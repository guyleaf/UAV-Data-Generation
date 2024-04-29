#!/usr/bin/env bash
set -e

pip install -U pip

pip install -U setuptools wheel

# change url according to your cuda
mamba install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

pip install -U -r requirements.txt
