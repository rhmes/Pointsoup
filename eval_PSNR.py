import os
import csv
import torch
import argparse
import kit.op as op
import open3d as o3d
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(
    prog='eval_PSNR.py',
    description='Eval Geometry PSNR.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', type=str, help='Glob pattern to load point clouds.', default='./data/example_pcs/conferenceRoom_1.ply')
parser.add_argument('--compressed_path', type=str, help='Path to save compressed files.', default='./data/compressed/')
parser.add_argument('--decompressed_path', type=str, help='Path to save decompressed files.', default='./data/decompressed/')
parser.add_argument('--resolution', type=int, help='Point cloud resolution (peak signal).', default=1023)
parser.add_argument('--csv_dir', type=str, help='Directory to save CSV results.', default='./csv/')

# Parse Arguments
args = parser.parse_args()

import numpy as np
from scipy.spatial import cKDTree

import os

def bits_per_point(file_name, num_points):
    """
    file_name: name of the compressed file (without path)
    num_points: total number of points in point cloud
    """
    # Bits per point (bpp) metric
    # Sum sizes of .b.bin, .h.bin, .s.bin as in compress.py
    bones_file = os.path.join(args.compressed_path, file_name + '.b.bin')
    head_file = os.path.join(args.compressed_path, file_name + '.h.bin')
    skin_file = os.path.join(args.compressed_path, file_name + '.s.bin')
    total_bits = 0
    total_bits = op.get_file_size_in_bits(bones_file) + op.get_file_size_in_bits(skin_file) + op.get_file_size_in_bits(head_file)
    bpp = total_bits/num_points
    return bpp

def chamfer_distance(pc1, pc2, norm=False):
    """
    pc1: (N, 3) numpy array
    pc2: (M, 3) numpy array
    returns: Chamfer Distance (float)
    """
    # Normalize point clouds with respect to their reference
    if norm:
        mean = pc1.mean(axis=0)
        scale = np.max(np.linalg.norm(pc1 - mean, axis=1))
        pc1 = (pc1 - mean) / scale
        pc2 = (pc2 - mean) / scale

    # Compute all pairwise distances
    pc1 = np.asarray(pc1)
    pc2 = np.asarray(pc2)
    # (N, M, 3) -> (N, M)
    diff = pc1[:, None, :] - pc2[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    # For each point in pc1, find min distance to pc2
    min_dist1 = np.min(dist_matrix, axis=1)
    # For each point in pc2, find min distance to pc1
    min_dist2 = np.min(dist_matrix, axis=0)
    cd = np.mean(min_dist1**2) + np.mean(min_dist2**2)
    return cd

def process(input_f):
    base_name = os.path.split(input_f)[-1]
    dec_f = os.path.join(args.decompressed_path, base_name + '.bin.ply')

    # Load point clouds
    pc_ref = o3d.io.read_point_cloud(input_f)
    pc_dec = o3d.io.read_point_cloud(dec_f)

    # Convert to numpy arrays
    ref_points = np.asarray(pc_ref.points)
    dec_points = np.asarray(pc_dec.points)

    # Compute P2P distances (dec to ref)
    distance = pc_dec.compute_point_cloud_distance(pc_ref)
    distance = np.asarray(distance)
    mse = np.mean(distance ** 2) + 1e-10  # Avoid division by zero
    # D1 PSNR
    peak_signal = args.resolution
    d1_psnr = 10 * np.log10((peak_signal ** 2) / mse) if mse > 0 else float('inf')

    # Chamfer Distance 
    chamfer_dist = chamfer_distance(ref_points, dec_points, norm=True)
    
    # bitrate
    bpp = bits_per_point(base_name, ref_points.shape[0])
    # Save all metrics
    result = {
        'filename': base_name,
        'd1_psnr': d1_psnr,
        'chamfer_distance': chamfer_dist,
        'bpp': bpp
    }
    # print(f"File: {base_name}, P2P(dec2ref): {result['p2p_dec2ref']:.6f}, P2P(ref2dec): {result['p2p_ref2dec']:.6f}, Chamfer: {chamfer_dist:.6f}, D1 PSNR: {d1_psnr:.3f}")
    return result


def evaluate(files):
    f_len = len(files)
    with Pool(4) as p:
        results = list(tqdm(p.imap(process, files), total=f_len))

    # Save to CSV in 'csv' folder, prefix with decompressed folder name
    os.makedirs(args.csv_dir, exist_ok=True)
    prefix = os.path.basename(os.path.normpath(args.decompressed_path))
    csv_filename = f"{prefix}_psnr_results.csv"
    csv_path = os.path.join(args.csv_dir, csv_filename)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'd1_psnr', 'chamfer_distance', 'bpp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Print average metrics
    d1_psnr_list = [r['d1_psnr'] for r in results]
    chamfer_list = [r['chamfer_distance'] for r in results]
    bpp_list = [r['bpp'] for r in results]
    print('Avg. D1 PSNR:', round(np.mean(d1_psnr_list), 3))
    print('Avg. Chamfer Distance:', round(np.mean(chamfer_list), 6))
    print('Avg. BPP:', round(np.mean(bpp_list), 6))

if __name__ == '__main__':
    # Input files
    files = np.array(glob(args.input_glob))
    print(f"Evaluating {len(files)} point clouds...")
    # Evaluate Point Clouds
    evaluate(files)
