#!/bin/bash
# Script to create and activate the Pointsoup environment based on device (MPS, CUDA, or CPU)

# Unified environment name
ENV_NAME="pointsoup"

# Detect device
if [[ $(uname -s) == "Darwin" ]] && [[ $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    DEVICE="MPS"
elif command -v nvidia-smi &> /dev/null; then
    DEVICE="CUDA"
else
    DEVICE="CPU"
fi

echo "Detected device: $DEVICE"
echo "Creating conda environment: $ENV_NAME"

YML_FILE="environment/environment.yml"
if [[ $DEVICE == "MPS" || $DEVICE == "CPU" ]]; then
    YML_FILE="environment/environment_cpu.yml"
fi

conda env create -f $YML_FILE || { echo "Failed to create conda environment."; exit 1; }
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME || { echo "Failed to activate conda environment."; exit 1; }
echo "Environment $ENV_NAME created and all packages installed."
