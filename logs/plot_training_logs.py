import matplotlib.pyplot as plt
import os
import numpy as np

def parse_log(log_path):
    steps, losses, dists, rates = [], [], [], []
    with open(log_path, 'r') as f:
        for line in f:
            # Only parse lines with all required fields
            if 'Step:' in line and 'Loss:' in line and 'Dist:' in line and 'Rate:' in line:
                parts = line.strip().split('|')
                try:
                    step = int(parts[0].split(':')[1].strip())
                    loss = float(parts[1].split(':')[1].strip())
                    dist = float(parts[2].split(':')[1].strip())
                    rate = float(parts[3].split(':')[1].strip())

                    steps.append(np.round(step))
                    losses.append(np.float32(loss))
                    dists.append(np.float32(dist))
                    rates.append(np.float32(rate))
                except Exception:
                    continue
    return steps, losses, dists, rates

def moving_average(arr, window=20):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

# Paths to your log files
log_pointsoup = './Pointsoup-training-log.txt'
log_pointsoup_sa = './Pointsoup-sa-training-log.txt'

print("Log files:")
print(f" - {log_pointsoup}")
print(f" - {log_pointsoup_sa}")
if not os.path.exists(log_pointsoup) or not os.path.exists(log_pointsoup_sa):
    print("Log files not found. Please check the paths.")
    exit(1)

steps1, losses1, dists1, rates1 = parse_log(log_pointsoup)
steps2, losses2, dists2, rates2 = parse_log(log_pointsoup_sa)

# Apply moving average
window = 20
ma_losses1 = moving_average(losses1, window)
ma_losses2 = moving_average(losses2, window)
ma_dists1 = moving_average(dists1, window)
ma_dists2 = moving_average(dists2, window)
ma_rates1 = moving_average(rates1, window)
ma_rates2 = moving_average(rates2, window)
ma_steps1 = steps1[window-1:]
ma_steps2 = steps2[window-1:]

fig, axs = plt.subplots(1, 3, figsize=(14, 3), sharex=True)

# Define styles
colors = {'pointsoup': 'tab:blue', 'pointsoup-sa': 'tab:red'}
linestyles = {'pointsoup': '--', 'pointsoup-sa': '-'}

# Loss plot
axs[0].plot(ma_steps1, ma_losses1, label='pointsoup', 
            color=colors['pointsoup'], linestyle=linestyles['pointsoup'])
axs[0].plot(ma_steps2, ma_losses2, label='pointsoup-sa', 
            color=colors['pointsoup-sa'], linestyle=linestyles['pointsoup-sa'])
axs[0].set_xlabel('Step')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# Chamfer Distance plot
axs[1].plot(ma_steps1, ma_dists1, label='pointsoup', 
            color=colors['pointsoup'], linestyle=linestyles['pointsoup'])
axs[1].plot(ma_steps2, ma_dists2, label='pointsoup-sa', 
            color=colors['pointsoup-sa'], linestyle=linestyles['pointsoup-sa'])
axs[1].set_xlabel('Step')
axs[1].set_ylabel('Chamfer Distance')
axs[1].set_title('Chamfer Distance')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

# Rate plot
axs[2].plot(ma_steps1, ma_rates1, label='pointsoup', 
            color=colors['pointsoup'], linestyle=linestyles['pointsoup'])
axs[2].plot(ma_steps2, ma_rates2, label='pointsoup-sa', 
            color=colors['pointsoup-sa'], linestyle=linestyles['pointsoup-sa'])
axs[2].set_xlabel('Step')
axs[2].set_ylabel('Rate')
axs[2].set_title('Training Rate')
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.6)

# Layout & style
plt.tight_layout()
plt.savefig('training_log_curves.png')
print('Saved plot to training_log_curves.png')
plt.show()