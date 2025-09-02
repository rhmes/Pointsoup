import os
import csv
import torch
import network
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import kit.io as io
import kit.utils as utils
import torch.utils.data as Data
from pytorch3d.loss import chamfer_distance

from scipy.spatial import cKDTree

# def chamfer_distance(pc1, pc2):
#     tree1 = cKDTree(pc1)
#     tree2 = cKDTree(pc2)
#     dist1, _ = tree1.query(pc2)
#     dist2, _ = tree2.query(pc1)
#     return np.mean(dist1) + np.mean(dist2)

def psnr(ref, rec):
    mse = np.mean((ref - rec) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1.0  # assuming normalized
    return 20 * np.log10(max_val / np.sqrt(mse))

parser = argparse.ArgumentParser(
    prog='evaluate_simple.py',
    description='Run model end-to-end and calculate PSNR and Chamfer Distance.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input_glob', type=str, default='./data/example_pc_1023/*.ply', help='Glob pattern to load point clouds.')
parser.add_argument('--model_load_path', type=str, help='Directory where to load trained models.', default=f'./model/exp/ckpt.pt')
parser.add_argument('--local_window_size', type=int, help='Local window size.', default=128)
parser.add_argument('--dilated_window_size', type=int, help='Dilated window size. (Same value with train.py)', default=8)
parser.add_argument('--channel', type=int, help='Network channel. (Same value with train.py)', default=128)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel. (Same value with train.py)', default=16)
parser.add_argument('--model_type', help='Model type (pointsoup or pointsoup_sa).', default='pointsoup')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = network.model(args, model_type=args.model_type)
model.load_state_dict(torch.load(args.model_load_path, map_location=device))
# model = utils.load_legacy_checkpoint(model, args.model_load_path, device=device)
# model = torch.compile(model)
model.to(device).eval()

def evaluate(loader):
    results = []
    with torch.no_grad():
        print("Evaluating point clouds...")
        for file in tqdm(loader):
            pc = io.read_point_clouds(file)[0]
            pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(device)
            # Forward through model
            rec, bitrate = model(pc, args.local_window_size)
            rec_np = rec[0].squeeze(0).cpu().numpy()
            print("pc:", pc.shape, "rec:", rec.shape, "rec[0]:", rec[0].shape)
            print("rec:", rec_np.shape)
            # Metrics
            # chamfer = chamfer_distance(pc, rec_np)
            chamfer, _ = chamfer_distance(pc, rec_np)
            psnr_val = psnr(pc, rec_np)
            results.append((os.path.basename(file), chamfer, psnr_val, bitrate))
            print(f"{os.path.basename(file)} | Chamfer: {chamfer:.6f} | PSNR: {psnr_val:.2f} | Bitrate: {bitrate:.2f}")
    return results

if __name__ == "__main__":
    files = np.array(glob(args.input_glob, recursive=True))[:10000]
        
    # Pointcloud data loader
    loader = Data.DataLoader(
        dataset = files,
        batch_size = 1,
        shuffle = True,
    )
    results = evaluate(loader)
    # Optionally save results to CSV
    with open('eval_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'chamfer', 'psnr', 'bitrate'])
        writer.writerows(results)
    print("Results saved to eval_results.csv")
