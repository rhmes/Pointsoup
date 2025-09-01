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
    prog='decompress.py',
    description='Deompress point clouds.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--compressed_path', type=str, help='Path to save .bin files.', default='./data/compressed/')
parser.add_argument('--decompressed_path', type=str, help='Path to save decompressed files.', default='./data/decompressed/')
parser.add_argument('--model_load_path', type=str, help='Directory where to load trained models.', default=f'./model/exp/ckpt.pt')
parser.add_argument('--tmc_path', type=str, help='TMC to compress bone points.', default='./tmc3-xos')

parser.add_argument('--verbose', type=bool, help='Print compression details.', default=False)

parser.add_argument('--dilated_window_size', type=int, help='Dilated window size. (Same value with train.py)', default=8)
parser.add_argument('--channel', type=int, help='Network channel. (Same value with train.py)', default=128)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel. (Same value with train.py)', default=16)
parser.add_argument('--model_type', help='Model type (pointsoup or pointsoup_sa).', default='pointsoup')

args = parser.parse_args()

# Create decompressed directory if not exists
if not os.path.exists(args.decompressed_path):
    os.makedirs(args.decompressed_path)

# Load model
model = network.model(args, model_type=args.model_type)
# Load checkpoint
model = utils.load_legacy_checkpoint(model, args.model_load_path, device=device)
model = torch.compile(model)
model = model.to(device).eval()


# warm up our model, since the first step of the model is very slow
model(torch.randn(1, 1024, 3).to(device), 128)

compressed_bones_path_ls = list(glob(os.path.join(args.compressed_path, '*.b.bin')))

time_recoder = op.Recoder()
ticker = op.Ticker()

with torch.no_grad():
    for compressed_bone_path in tqdm(compressed_bones_path_ls):

        filename_w_ext = os.path.split(compressed_bone_path[:-6])[-1]
        head_path, bones_path, skin_path, cache_path = utils.compressed_files(filename_w_ext, args.compressed_path)
        decompressed_path = os.path.join(args.decompressed_path, filename_w_ext+'.bin.ply')


        ######################################################
        ################## Entropy Modeling ##################
        ######################################################

        ############## 🚩 Bone Decompression ##############
        # (io time is omitted since the tmc process can be done in RAM in practial applications)
    
        bone_dec_time = op.tmc_decompress(args.tmc_path, bones_path, cache_path)
        rec_bones = torch.tensor(io.read_point_cloud(cache_path)).float().to(device)
        M = rec_bones.shape[0]

        ticker.set_time('TMCDecTime', bone_dec_time) # 🕒 ✔️
        if args.verbose:
            print('[TMC] Dec:', ticker.get_time('TMCDecTime'), 's')
            
        ############## 🚩 DW-Build ##############
            
        ticker.start_count('DWBuild') # 🕒 ⏳

        dilated_idx, dilated_windows = model.dw_build(rec_bones)

        ticker.end_count('DWBuild') # 🕒 ✔️
        if args.verbose:
            print('[DWBuild]:', ticker.get_time('DWBuild'), 's')
        
        ############## 🚩 DWEM ##############
            
        ticker.start_count('DWEM') # 🕒 ⏳

        mu, sigma = model.dwem(dilated_windows)

        ticker.end_count('DWEM') # 🕒 ✔️
        if args.verbose:
            print('[DWEM]:', ticker.get_time('DWEM'), 's')

        ############## 🚩 Arithmetic Decoding ##############
            
        # get vlaue boundries from head file
        with open(head_path, 'rb') as fin:
            local_window_size = np.frombuffer(fin.read(2), dtype=np.uint16)[0]
            min_v_value = np.frombuffer(fin.read(2), dtype=np.int16)[0]
            max_v_value = np.frombuffer(fin.read(2), dtype=np.int16)[0]

        # get skin bit stream
        with open(skin_path, 'rb') as fin:
            bytestream = fin.read()
            
        ticker.start_count('AD') # 🕒 ⏳

        quantized_compact_fea = torchac.decode_int16_normalized_cdf(
            op._convert_to_int_and_normalize(op.get_cdf_min_max_v(mu-min_v_value, sigma, L=max_v_value-min_v_value+1), needs_normalization=True).cpu(), 
            bytestream
        ) + min_v_value
        quantized_compact_fea = quantized_compact_fea.float().to(device)

        ticker.end_count('AD') # 🕒 ✔️
        if args.verbose:
            print('[AD]:', ticker.get_time('AD'), 's')
        

        ######################################################
        ######################## DWUS ########################
        ######################################################
        
        ticker.start_count('DWUS') # 🕒 ⏳

        # feature stretching
        rec_skin_fea = model.fea_stretch(quantized_compact_fea)
        rec_batch_x = model.dwus(rec_skin_fea, rec_bones, dilated_windows, dilated_idx, local_window_size)

        ticker.end_count('DWUS') # 🕒 ✔️
        if args.verbose:
            print('[DWUS]:', ticker.get_time('DWUS'), 's')

        # save rec point cloud
        io.save_point_cloud(rec_batch_x[0], decompressed_path)

        dec_time = ticker.dump_sum()
        time_recoder.update(dec_time)
        if args.verbose:
            print(f'{filename_w_ext} done. Decoding time: {dec_time}s.')

        # remove cache file
        # but it is ok not to clean it up, it won't affect the code running...
        output = subprocess.check_output(f'rm {cache_path}', shell=True, stderr=subprocess.STDOUT)

print(f'Done. Avg. Decoding time: {time_recoder.dump_avg(precision=3)}s.')
