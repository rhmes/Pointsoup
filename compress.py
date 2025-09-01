import os
import random
import argparse
import subprocess

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchac

import kit.io as io
import kit.op as op
import kit.utils as utils

import network
import warnings
import multiprocessing

# Set start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore")

seed = 11
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(
    prog='compress.py',
    description='Compress point clouds.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', type=str, help='Glob pattern to load point clouds.', default='./data/example_pcs/*.ply')
parser.add_argument('--compressed_path', type=str, help='Path to save .bin files.', default='./data/compressed/')
parser.add_argument('--model_load_path', type=str, help='Directory where to load trained models.', default=f'./model/exp/ckpt.pt')
parser.add_argument('--tmc_path', type=str, help='TMC to compress bone points.', default='./tmc3')

parser.add_argument('--local_window_size', type=int, help='Local window size.', default=128)
parser.add_argument('--verbose', type=bool, help='Print compression details.', default=False)

parser.add_argument('--dilated_window_size', type=int, help='Dilated window size. (Same value with train.py)', default=8)
parser.add_argument('--channel', type=int, help='Network channel. (Same value with train.py)', default=128)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel. (Same value with train.py)', default=16)
parser.add_argument('--model_type', help='Model type (pointsoup or pointsoup_sa).', default='pointsoup')

args = parser.parse_args()

# Create compressed directory if not exists
if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)

# Load model
model = network.model(args, model_type=args.model_type)
# Load checkpoint
model = utils.load_legacy_checkpoint(model, args.model_load_path, device=device)
model = torch.compile(model)
model = model.to(device).eval()

# warm up our model, since the first step of the model is very slow
model(torch.randn(1, 1024, 3).to(device), 128)

files = np.array(glob(args.input_glob, recursive=True))

time_recoder, bpp_recoder = op.Recoder(), op.Recoder()
ticker = op.Ticker()

with torch.no_grad():
    for file_path in tqdm(files):

        pc = io.read_point_cloud(file_path)
        batch_x = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(device)
        N = batch_x.shape[1]

        filename_w_ext = os.path.split(file_path)[-1]
        head_path, bones_path, skin_path, cache_path = utils.compressed_files(filename_w_ext, args.compressed_path)

        if args.verbose:
            print('*'*30)
            print('Processing:', filename_w_ext, f'({N} points)')

        ######################################################
        ######################## AWDS ########################
        ######################################################

        ############## 🚩 SamplingAndQuery ##############

        ticker.start_count('SamplingAndQuery') # 🕒 ⏳

        bones, local_windows = op.SamplingAndQuery(batch_x, K=args.local_window_size)
        # we get bones: (M, 3); local windows: (M, K, 3)

        ticker.end_count('SamplingAndQuery') # 🕒 ✔️
        if args.verbose:
            print('[AWDS] Sampling and query:', ticker.get_time('SamplingAndQuery'), 's')

        ############## 🚩 Bone Compression ##############
        # (io time is omitted since the tmc process can be done in RAM in practial applications)
        io.save_point_cloud(bones, cache_path)
        bone_steam_size, bone_enc_time = op.tmc_compress(args.tmc_path, cache_path, bones_path)
        bone_dec_time = op.tmc_decompress(args.tmc_path, bones_path, cache_path)
        rec_bones = torch.tensor(io.read_point_cloud(cache_path)).float().to(device) # -> (M, 3)
        
        ticker.set_time('TMCEncTime', bone_enc_time) # 🕒 ✔️
        ticker.set_time('TMCDecTime', bone_dec_time) # 🕒 ✔️
        if args.verbose:
            print('[TMC] Enc:', ticker.get_time('TMCEncTime'), 's')
            print('[TMC] Dec:', ticker.get_time('TMCDecTime'), 's')

        ############## 🚩 Encoding ##############
        ############## 🚩 Adaptive Aligning ##############
        
        ticker.start_count('Aligning') # 🕒 ⏳

        # we have to re-sort the bones&windows since decoded bones has different point order
        # otherwise the bones and window cannot one-by-one match
        cloest_idx = op.reorder(bones, rec_bones)
        bones, local_windows = bones[cloest_idx], local_windows[cloest_idx]
        # we get bones: (M, 3); local windows: (M, K, 3)
        aligned_windows = op.AdaptiveAligning(local_windows, bones)
        # aligned_windows: (M, K, 3)

        ticker.end_count('Aligning') # 🕒 ✔️
        if args.verbose:
            print('[AWDS] Aligning:', ticker.get_time('Aligning'), 's')

        ############## 🚩 Attention-based Aggregation ##############

        ticker.start_count('Aggregation') # 🕒 ⏳

        # due to the GPU memory limit
        # the attention block cannot deal with all the windows in one shot
        # we group the windows into mini batches
        mini_batch_size = 524288 // args.local_window_size
        cursor = 0
        skin_fea_ls = []
        while cursor < aligned_windows.shape[0]:
            skin_fea_item = model.encoder.awds.mini_emb(aligned_windows[cursor:cursor+mini_batch_size])
            skin_fea_item = model.encoder.awds.pt(aligned_windows[cursor:cursor+mini_batch_size], skin_fea_item)
            skin_fea_ls.append(skin_fea_item)
            cursor = cursor + mini_batch_size
        skin_fea = torch.cat(skin_fea_ls, dim=0)
        # get skin_fea: (M, C)

        ticker.end_count('Aggregation') # 🕒 ✔️
        if args.verbose:
            print('[AWDS] Aggregation:', ticker.get_time('Aggregation'), 's')

        ############## 🚩 Global Correlation & Simple Attention (Pointsoup-SA only) ##############
        
        if args.model_type == 'pointsoup_sa':
            ticker.start_count('Global_Corr') # 🕒 ⏳

            skin_fea = model.encoder.pt_block(skin_fea)

            ticker.end_count('Global_Corr') # 🕒 ✔️
            if args.verbose:
                print('[Global_Corr]:', ticker.get_time('Global_Corr'), 's')

        ######################################################
        ################## Entropy Modeling ##################
        ######################################################
            
        ############## 🚩 DW-Build ##############
            
        ticker.start_count('DWBuild') # 🕒 ⏳

        dilated_idx, dilated_windows = model.entropy_model.dw_build(rec_bones)

        ticker.end_count('DWBuild') # 🕒 ✔️
        if args.verbose:
            print('[DWBuild]:', ticker.get_time('DWBuild'), 's')

        ############## 🚩 DWEM ##############
            
        ticker.start_count('DWEM') # 🕒 ⏳

        mu, sigma = model.entropy_model.dwem(dilated_windows)

        ticker.end_count('DWEM') # 🕒 ✔️
        if args.verbose:
            print('[DWEM]:', ticker.get_time('DWEM'), 's')

        ############## 🚩 Feature Squeezeing & Quantization ##############
            
        ticker.start_count('Squeezeing_Q') # 🕒 ⏳
            
        compact_fea = model.encoder.fea_squeeze(skin_fea)
        quantized_compact_fea = torch.round(compact_fea)
        # get quantized_compact_fea: (M, c)

        ticker.end_count('Squeezeing_Q') # 🕒 ✔️
        if args.verbose:
            print('[Squeezeing]+[Q]:', ticker.get_time('Squeezeing_Q'), 's')

        ############## 🚩 Arithmetic Encoding ##############
            
        ticker.start_count('AE') # 🕒 ⏳
        min_v_value, max_v_value = quantized_compact_fea.min().to(torch.int16), quantized_compact_fea.max().to(torch.int16)
        bytestream = torchac.encode_int16_normalized_cdf(
            op._convert_to_int_and_normalize(op.get_cdf_min_max_v(mu-min_v_value, sigma, L=max_v_value-min_v_value+1), needs_normalization=True).cpu(), 
            (quantized_compact_fea-min_v_value).cpu().to(torch.int16)
        )

        ticker.end_count('AE') # 🕒 ✔️
        if args.verbose:
            print('[AE]:', ticker.get_time('AE'), 's')

        ############## Saving ##############
        with open(head_path, 'wb') as fout:
            fout.write(np.array(args.local_window_size, dtype=np.uint16).tobytes())
            fout.write(np.array(min_v_value.item(), dtype=np.int16).tobytes())
            fout.write(np.array(max_v_value.item(), dtype=np.int16).tobytes())

        with open(skin_path, 'wb') as fin:
            fin.write(bytestream)

        ############## Calc Enc Time and Bitrate ##############

        # Calc Bpp
        total_bits = op.get_file_size_in_bits(bones_path) + op.get_file_size_in_bits(skin_path) + op.get_file_size_in_bits(head_path)
        bpp = total_bits/N

        # Calc Enc Time
        enc_time = ticker.dump_sum()
        
        # Add to recoder
        time_recoder.update(enc_time)
        bpp_recoder.update(bpp)

        if args.verbose:
            print(f'{filename_w_ext} done. Encoding time: {enc_time}s | Bpp: {round(bpp, 3)}.')

        # remove cache file
        # but it is ok not to clean it up, it won't affect the code running...
        output = subprocess.check_output(f'rm {cache_path}', shell=True, stderr=subprocess.STDOUT)
    
print(f'Done. Avg. Encoding time: {time_recoder.dump_avg(precision=3)} | Bpp: {bpp_recoder.dump_avg(precision=3)}')
