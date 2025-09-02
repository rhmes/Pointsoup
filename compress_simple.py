import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import kit.io as io
import kit.utils as utils
import network

parser = argparse.ArgumentParser(
    prog='compress_simple.py',
    description='Compress point clouds using model.encoder directly.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input_glob', type=str, default='./data/example_pcs/*.ply', help='Glob pattern to load point clouds.')
parser.add_argument('--compressed_path', type=str, default='./data/compressed/', help='Path to save .bin files.')
parser.add_argument('--model_load_path', type=str, default='./model/exp/ckpt.pt', help='Path to trained model.')
parser.add_argument('--local_window_size', type=int, help='Local window size.', default=128)

parser.add_argument('--dilated_window_size', type=int, help='Dilated window size. (Same value with train.py)', default=8)
parser.add_argument('--channel', type=int, help='Network channel. (Same value with train.py)', default=128)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel. (Same value with train.py)', default=16)
parser.add_argument('--model_type', type=str, default='pointsoup', help='Model type.')
args = parser.parse_args()

if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = network.model(args, model_type=args.model_type)
model = utils.load_legacy_checkpoint(model, args.model_load_path, device=device)
model = torch.compile(model)
model = model.to(device).eval()

files = np.array(glob(args.input_glob, recursive=True))
with torch.no_grad():
    bpp_list = []
    for file_path in tqdm(files):
        pc = io.read_point_cloud(file_path)
        batch_x = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(device)
        filename_w_ext = os.path.split(file_path)[-1]
        # Simulate bones, skin, head outputs from encoder
        encoded = model.encoder(batch_x, args.local_window_size)
        # For demonstration, split encoded tensor into three parts
        # (In real use, replace with actual outputs)
        bones = encoded[0].cpu()
        features = encoded[1].cpu()

        # Save as raw binary files
        bones_path = os.path.join(args.compressed_path, filename_w_ext + '.bones.bin')
        features_path = os.path.join(args.compressed_path, filename_w_ext + '.features.bin')
        bones.numpy().tofile(bones_path)
        features.numpy().tofile(features_path)

        # Bitrate calculation (same as compress.py)
        num_points = batch_x.shape[1]
        import kit.op as op
        total_bits = op.get_file_size_in_bits(bones_path) + op.get_file_size_in_bits(features_path)
        bpp = total_bits / num_points
        bpp_list.append(bpp)
    print(f"Average bpp: {sum(bpp_list) / len(bpp_list):.3f}")
    # Save bpp_list for analysis
    np.savez_compressed(os.path.join(args.compressed_path, "bpp_stats.npz"), bpp=bpp_list)
print("Done.")
