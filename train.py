import os
import random
import argparse

import numpy as np
from glob import glob
from datetime import datetime

import torch
import torch.utils.data as Data
from pytorch3d.loss import chamfer_distance

import kit.io as io
import kit.op as op
import network

from tqdm import tqdm
import warnings
import multiprocessing

# Set start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)
# Ignore warnings
warnings.filterwarnings("ignore")

# Set random seeds
seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set torch device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--model_save_folder', help='Directory where to save trained models.', default=f'./model/exp/')
parser.add_argument('--train_glob', help='Glob pattern to load point clouds.', default='/mnt/hdd/datasets_yk/ShapeNet/ShapeNet_pc_01_8192p_colorful/train/*.ply')

parser.add_argument('--local_window_size', type=int, help='Local window size $K$.', default=128)
parser.add_argument('--dilated_window_size', type=int, help='Dilated window size $k$.', default=8)
parser.add_argument('--channel', type=int, help='Network channel.', default=128)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel.', default=16)
parser.add_argument('--λ_R', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-04)

parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion tradeoff at x steps.', default=5000)
parser.add_argument('--batch_size', type=int, help='Batch size (must be 1).', default=1)
parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate at x step.', default=[70000, 120000])
parser.add_argument('--max_step', type=int, help='Train up to this number of steps.', default=140000)
parser.add_argument('--step_window', type=int, help='Step window for stats and checkpoints.', default=200)
parser.add_argument('--model_type', help='Model type (pointsoup or pointsoup_sa).', default='pointsoup')

# Parse Arguments
args = parser.parse_args()

# Save model
def save_model(model, optimizer, step):
    torch.save(model.state_dict(), os.path.join(args.model_save_folder, 'ckpt.pt'))
    torch.save(optimizer.state_dict(), os.path.join(args.model_save_folder, 'optimizer.pt'))
    with open(os.path.join(args.model_save_folder, 'step.txt'), 'w') as f:
        f.write(str(step))

# Load model
def load_model(model, optimizer):
    step = 0
    ckpt_path = os.path.join(args.model_save_folder, 'ckpt.pt')
    op_path = os.path.join(args.model_save_folder, 'optimizer.pt')
    st_path = os.path.join(args.model_save_folder, 'step.txt')
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

# Init model
def init_model(model_type="pointsoup"):
    # Select model type:  -- pointsoup -> Pointsoup 
    #                     -- pointsoup_sa -> Pointsoup Self-Attention
    if model_type == "pointsoup_sa":
        model = network.PointsoupSelfAttention(k=args.dilated_window_size,
                            channel=args.channel, 
                            bottleneck_channel=args.bottleneck_channel)
        print("[TRAIN] Using \033[1;32mPointsoup Self-Attention\033[0m model.")
    else:
        model = network.Pointsoup(k=args.dilated_window_size,
                            channel=args.channel, 
                            bottleneck_channel=args.bottleneck_channel)
        print("[TRAIN] Using \033[1;34mPointsoup\033[0m model.")
    return model

# Training function
def train(loader):
    # Initialize model and optimizer
    model = init_model(model_type=args.model_type)
    # Move model to device and set to training mode
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load latest checkpoint if exists, update global_step if exists
    global_step = 0
    global_step = load_model(model, optimizer)

    # Restore epoch number
    start = global_step // args.max_step 
    cd_recoder, bpp_recoder, loss_recoder = op.Recoder(), op.Recoder(), op.Recoder()
    
    pbar = tqdm(desc="Train", initial=global_step, total=args.max_step, unit="step")
    for epoch in range(start, 9999):
        print(f"[TRAIN] {datetime.now()}")
        for batch_x in loader:
            batch_x = batch_x.to(device)

            rec_batch_x, bitrate = model(batch_x, args.local_window_size)

            # Get Loss
            chamfer_dist, _ = chamfer_distance(rec_batch_x, batch_x)
            loss = chamfer_dist
            if global_step > args.rate_loss_enable_step:
                loss += args.λ_R * bitrate
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            # Update recorders
            cd_recoder.update(chamfer_dist.item())
            bpp_recoder.update(bitrate.item())
            loss_recoder.update(loss.item())
            # Update progress bar postfix
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })
            # Print training stats, dump checkpoints per step-window (default: 200)
            if global_step % args.step_window == 0:
                print(f'[Epoch {epoch}] Step {global_step} | '
                    f'Loss: {loss_recoder.dump_avg():} | '
                    f'Dist: {cd_recoder.dump_avg()} | '
                    f'Rate: {bpp_recoder.dump_avg()}')
                # Save model
                save_model(model, optimizer, global_step)

            # Learning Rate Decay
            if global_step in args.lr_decay_steps:
                args.lr = args.lr * args.lr_decay
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
                print(f'Learning rate decay triggered at step {global_step}, LR is setting to {args.lr}.')
            
            pbar.update(1)
            # Stop training if global_step exceeds max_step
            if global_step > args.max_step:
                break
        # Stop training with max_step
        if global_step > args.max_step:
            break
    pbar.close()
    print('[TRAIN] Training Finished!')

if __name__ == '__main__':
    # Create save-model directory
    if not os.path.exists(args.model_save_folder):
        os.makedirs(args.model_save_folder)
    # Read input point clouds
    files = np.array(glob(args.train_glob, recursive=True))[:10000]
    pcs = io.read_point_clouds(files)
    # Pointcloud data loader
    loader = Data.DataLoader(
        dataset = pcs,
        batch_size = args.batch_size,
        shuffle = True,
    )
    # Training process
    train(loader)
