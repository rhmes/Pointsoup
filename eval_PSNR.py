import os
import argparse

import open3d as o3d

import numpy as np
from glob import glob
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser(
    prog='eval_PSNR.py',
    description='Eval Geometry PSNR.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', type=str, help='Glob pattern to load point clouds.', default='./data/example_pcs/conferenceRoom_1.ply')
parser.add_argument('--decompressed_path', type=str, help='Path to save decompressed files.', default='./data/decompressed/')
parser.add_argument('--pcc_metric_path', type=str, help='Path for PccAppMetrics.', default='./PccAppMetrics')
parser.add_argument('--resolution', type=int, help='Point cloud resolution (peak signal).', default=1023)

# Parse Arguments
args = parser.parse_args()

def process(input_f):
    filename_w_ext = os.path.split(input_f)[-1]
    dec_f = os.path.join(args.decompressed_path, filename_w_ext+'.bin.ply')

    # Load point clouds
    pc_ref = o3d.io.read_point_cloud(input_f)
    pc_dec = o3d.io.read_point_cloud(dec_f)

    # Convert to numpy arrays
    ref_points = np.asarray(pc_ref.points)
    dec_points = np.asarray(pc_dec.points)

    # Compute distances from dec to ref
    distances = pc_dec.compute_point_cloud_distance(pc_ref)
    distances = np.array(distances)
    mse = np.mean(distances ** 2)
    peak_signal = args.resolution
    d1_psnr = 10 * np.log10((peak_signal ** 2) / mse) if mse > 0 else float('inf')

    return np.array([filename_w_ext, d1_psnr])

def evaluate(files): 
    f_len = len(files)
    with Pool(4) as p:
        arr = list(tqdm(p.imap(process, files), total=f_len))

    arr = np.array(arr)
    fnames, p2pPSNRs = arr[:, 0], arr[:, 1].astype(float)
        
    print('Avg. D1 PSNR:', round(p2pPSNRs.mean(), 3))

if __name__ == '__main__':
    # Input files
    files = np.array(glob(args.input_glob))
    # Evaluate Point Clouds
    evaluate(files)

    pass