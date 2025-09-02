import os
import csv
import torch
import argparse
import kit.op as op
import open3d as o3d
import numpy as np
from glob import glob
from tqdm import tqdm

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

def chamfer_distance(pc_ref, pc_dec, norm=False):
    """
    pc1: (N, 3) numpy array
    pc2: (M, 3) numpy array
    returns: Chamfer Distance (float)
    """
    # Normalize point clouds using min-max scaling
    # Convert to numpy arrays
    pc1 = np.asarray(pc_ref.points)
    pc2 = np.asarray(pc_dec.points)
    if norm:
        min_xyz = np.min(pc1, axis=0)
        max_xyz = np.max(pc1, axis=0)
        range_xyz = max_xyz - min_xyz
        range_xyz[range_xyz == 0] = np.mean(range_xyz)  # Avoid division by zero
        pc1 = (pc1 - min_xyz) / range_xyz
        pc2 = (pc2 - min_xyz) / range_xyz
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

def point_to_plane_distance(pc_dec, pc_ref):
    # Estimate normals for reference cloud
    pc_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    ref_points = np.asarray(pc_ref.points)
    ref_normals = np.asarray(pc_ref.normals)
    dec_points = np.asarray(pc_dec.points)
    tree = cKDTree(ref_points)
    distances, indices = tree.query(dec_points, k=1)
    nearest_normals = ref_normals[indices]
    nearest_points = ref_points[indices]
    vec = dec_points - nearest_points
    ptp_dist = np.abs(np.sum(vec * nearest_normals, axis=1))
    return ptp_dist

def get_d1_psnr(pc_dec, pc_ref, peak_signal):
    # Compute P2P distances (dec to ref)
    distance = pc_dec.compute_point_cloud_distance(pc_ref)
    distance = np.asarray(distance)
    mse = np.mean(distance ** 2) + 1e-10  # Avoid division by zero
    d1_psnr = 10 * np.log10((peak_signal ** 2) / mse) if mse > 0 else float('inf')
    return d1_psnr

def get_d2_psnr(pc_dec, pc_ref, peak_signal):
    ptp_distances = point_to_plane_distance(pc_dec, pc_ref)
    mse_d2 = np.mean(ptp_distances ** 2) + 1e-10
    d2_psnr = 10 * np.log10((peak_signal ** 2) / mse_d2) if mse_d2 > 0 else float('inf')
    return d2_psnr

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

# Process a single file and return metrics
def process(input_f):
    base_name = os.path.split(input_f)[-1]
    dec_f = os.path.join(args.decompressed_path, base_name + '.bin.ply')

    # Load point clouds
    pc_ref = o3d.io.read_point_cloud(input_f)
    pc_dec = o3d.io.read_point_cloud(dec_f)

    # D1/D2 PSNR (point-to-point / point-to-plane)
    peak_signal = args.resolution
    d1_psnr = get_d1_psnr(pc_dec, pc_ref, peak_signal)
    d2_psnr = get_d2_psnr(pc_dec, pc_ref, peak_signal)
    # Bitrate (bits per point)
    N = len(pc_ref.points)
    bpp = bits_per_point(base_name, N)
    # Chamfer Distance (uncomment if needed)
    # chamfer_dist = chamfer_distance(pc_ref, pc_dec, norm=True)

    # Save all metrics
    result = {
        'filename': base_name,
        'd1_psnr': d1_psnr,
        'd2_psnr': d2_psnr,
        'bpp': bpp
    }
    return result

# Dump metrics to csv-file
def dump_metrics(results):
    # Avg metrics
    d1_avg, d2_avg, bpp_avg = 0, 0, 0
    length = len(results)
    # Save to CSV in 'csv' folder, prefix with decompressed folder name
    os.makedirs(args.csv_dir, exist_ok=True)
    prefix = os.path.basename(os.path.normpath(args.decompressed_path))
    csv_filename = f"{prefix}_psnr_results.csv"
    csv_path = os.path.join(args.csv_dir, csv_filename)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'd1_psnr', 'd2_psnr', 'bpp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Accumulate average metrics
            d1_avg += row['d1_psnr']
            d2_avg += row['d2_psnr']
            bpp_avg += row['bpp']
            # Write new csv row
            writer.writerow(row)
        # Append average values to CSV
        d1_avg = round(d1_avg/length, 3) if length else 0
        d2_avg = round(d2_avg/length, 3) if length else 0
        bpp_avg = round(bpp_avg/length, 6) if length else 0
        avg_row = {
            'filename': 'AVERAGE',
            'd1_psnr': d1_avg,
            'd2_psnr': d2_avg,
            'bpp': bpp_avg
        }
        # Write average row
        writer.writerow(avg_row)
    # Print average metrics
    print('Avg. D1 PSNR (point-to-point):', d1_avg)
    print('Avg. D2 PSNR (point-to-plane):', d2_avg)
    print('Avg. BPP (bits per point):', bpp_avg)

# Evaluate Point Clouds
def evaluate(files):
    f_len = len(files)
    results = []
    for f in tqdm(files, total=f_len):
        try:
            results.append(process(f))
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
    # Filter out results
    filtered_results = [r for r in results if not (np.isnan(r['d1_psnr']) or np.isnan(r['d2_psnr']) or np.isnan(r['bpp']))]
    # Save to CSV
    dump_metrics(filtered_results)

if __name__ == '__main__':
    # Input files
    files = np.array(glob(args.input_glob))
    print(f"Evaluating {len(files)} point clouds...")
    # Evaluate Point Clouds
    evaluate(files)
