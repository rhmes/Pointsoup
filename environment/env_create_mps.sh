#!/bin/bash
# Script to create and activate the pointsoup-mps environment for Apple Silicon (MPS)

# Create environment with Python 3.10
conda create -y -n pointsoup-mps python=3.10
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pointsoup-mps

# Install PyTorch (MPS/CPU/GPU compatible)
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch

# Install PyTorch3D dependencies
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath

# Install torchac and other Python packages
pip install torchac ninja pandas matplotlib plyfile pyntcloud

echo "Environment 'pointsoup-mps' created and all packages installed."

# PyTorch3D is not available via pip/conda for MacOS. To install, build from source:
pip install git+https://github.com/facebookresearch/pytorch3d.git
