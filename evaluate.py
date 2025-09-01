import subprocess
import argparse

parser = argparse.ArgumentParser(description='Evaluate model performance.')
parser.add_argument('--input_glob', type=str, default='./data/example_pc_1023/*.ply', help='Glob pattern to load point clouds.')
parser.add_argument('--model_load_path', type=str, default='./model/exp/ckpt.pt', help='Path to trained model checkpoint.')
parser.add_argument('--compressed_path', type=str, default='./data/compressed-t/', help='Path to save .bin files.')
parser.add_argument('--decompressed_path', type=str, default='./data/decompressed-t/', help='Path to save decompressed files.')
parser.add_argument('--local_window_size', type=str, default='128', help='Local window size.')
parser.add_argument('--tmc_path', type=str, default='./tmc3-xos', help='Path to TMC executable.')
parser.add_argument('--resolution', type=str, default='1023', help='Point cloud resolution (peak signal).')
parser.add_argument('--verbose', type=str, default='False', help='Print details.')
parser.add_argument('--model_type', help='Model type (pointsoup or pointsoup_sa).', default='pointsoup')

args = parser.parse_args()

# Unified highlight color (bold gray)
BOLD = '\033[1m'
GRAY = f'{BOLD}\033[90m'
ENDC = '\033[0m'

# Step 1: Compression
print(f"\n{GRAY}========== Process 1/3: Compression =========={ENDC}")
subprocess.run([
    'python', 'compress.py',
    '--input_glob', args.input_glob,
    '--compressed_path', args.compressed_path,
    '--model_load_path', args.model_load_path,
    '--local_window_size', args.local_window_size,
    '--tmc_path', args.tmc_path,
    '--model_type', args.model_type,
    # '--verbose', args.verbose
], check=True)

# Step 2: Decompression
print(f"\n{GRAY}========== Process 2/3: Decompression =========={ENDC}")
subprocess.run([
    'python', 'decompress.py',
    '--model_load_path', args.model_load_path,
    '--compressed_path', args.compressed_path,
    '--decompressed_path', args.decompressed_path,
    '--tmc_path', args.tmc_path,
    '--model_type', args.model_type,
    # '--verbose', args.verbose
], check=True)

# Step 3: PSNR Evaluation
print(f"\n{GRAY}========== Process 3/3: PSNR Evaluation =========={ENDC}")
subprocess.run([
    'python', 'eval_PSNR.py',
    '--input_glob', args.input_glob,
    '--compressed_path', args.compressed_path,
    '--decompressed_path', args.decompressed_path,
    '--resolution', args.resolution
], check=True)

print(f"\n{GRAY}========== All steps completed =========={ENDC}")