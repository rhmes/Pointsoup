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
warnings.filterwarnings("ignore")

seed = 11
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

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

# Parse Arguments
args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    args.device = torch.device('cpu')

def save_model(model, optimizer, step):
    torch.save(model.state_dict(), os.path.join(args.model_save_folder, 'ckpt.pt'))
    torch.save(optimizer.state_dict(), os.path.join(args.model_save_folder, 'optimizer.pt'))
    with open(os.path.join(args.model_save_folder, 'step.txt'), 'w') as f:
        f.write(str(step))

def load_model(model, optimizer, step):
    ckpt_path = os.path.join(args.model_save_folder, 'ckpt.pt')
    op_path = os.path.join(args.model_save_folder, 'optimizer.pt')
    st_path = os.path.join(args.model_save_folder, 'step.txt')

    ckpt_exist = os.path.exists(ckpt_path)
    op_exist = os.path.exists(op_path)
    st_exist = os.path.exists(st_path)

    if ckpt_exist:
        print(f"Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        if op_exist:
            print(f"Loading optimizer from {op_path}")
            optimizer.load_state_dict(torch.load(op_path, map_location=args.device))
        if st_exist:
            with open(st_path, 'r') as f:
                step = int(f.read())

def train(loader):
    model = network.Pointsoup(k=args.dilated_window_size,
                            channel=args.channel, 
                            bottleneck_channel=args.bottleneck_channel)
    model = model.to(args.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    global_step = 0
    # Load latest checkpoint if exists
    load_model(model, optimizer, global_step)
    
    # Restore epoch number
    start = global_step // args.max_step 
    cd_recoder, bpp_recoder, loss_recoder = op.Recoder(), op.Recoder(), op.Recoder()

    pbar = tqdm(desc="Train", initial=global_step, total=args.max_step, unit="step")
    for epoch in range(start, 9999):
        print(datetime.now())
        for batch_x in loader:
            batch_x = batch_x.to(args.device)

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

            cd_recoder.update(chamfer_dist.item())
            bpp_recoder.update(bitrate.item())
            loss_recoder.update(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })
            if global_step % 200 == 0:
                print(f'[Epoch {epoch}] Step {global_step} | '
                  f'Loss: {loss_recoder.dump_avg():} | '
                  f'Dist: {cd_recoder.dump_avg()} | '
                  f'Rate: {bpp_recoder.dump_avg()}')

                # save model
                save_model(model, optimizer, global_step)

            # Learning Rate Decay
            if global_step in args.lr_decay_steps:
                args.lr = args.lr * args.lr_decay
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
                print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')
            
            pbar.update(1)
            if global_step > args.max_step:
                break
        if global_step > args.max_step:
            break
    pbar.close()
    print('Training Finished!')

if __name__ == '__main__':
    # Create Model Save Path
    if not os.path.exists(args.model_save_folder):
        os.makedirs(args.model_save_folder)

    files = np.array(glob(args.train_glob, recursive=True))[:10000]
    pcs = io.read_point_clouds(files)

    loader = Data.DataLoader(
        dataset = pcs,
        batch_size = args.batch_size,
        shuffle = True,
    )
    # Train loop
    train(loader)
