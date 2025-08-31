import multiprocessing
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from pyntcloud import PyntCloud

import torch

# Read point cloud
def read_point_cloud(filepath):
    pc = PyntCloud.from_file(filepath)
    pc = np.array(pc.points)[:, :3]
    return pc

# Read multiple point clouds
def read_point_clouds(file_path_list):
    print('loading point clouds...')
    with multiprocessing.Pool(4) as p:
        pcs = list(tqdm(p.imap(read_point_cloud, file_path_list, 32), total=len(file_path_list)))
    return pcs

# Save point cloud
def save_point_cloud(pc, path):
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    df = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(path)

# Save model 
def save_model(model, optimizer, step, model_path):
    torch.save(model.state_dict(), os.path.join(model_path, 'ckpt.pt'))
    torch.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
    with open(os.path.join(model_path, 'step.txt'), 'w') as f:
        f.write(str(step))

# Load model
def load_model(model, optimizer, model_path, device):
    step = 0
    ckpt_path = os.path.join(model_path, 'ckpt.pt')
    op_path = os.path.join(model_path, 'optimizer.pt')
    st_path = os.path.join(model_path, 'step.txt')
    # Check if checkpoint exists
    if os.path.exists(ckpt_path):
        print(f"[TRAIN] Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        # Check if optimizer exists
        if os.path.exists(op_path):
            print(f"[TRAIN] Loading optimizer from {op_path}")
            optimizer.load_state_dict(torch.load(op_path, map_location=device))
        # Check if step exists
        if os.path.exists(st_path):
            with open(st_path, 'r') as f:
                step = int(f.read())
                print(f"[TRAIN] Resuming from step {step}")
    return step
