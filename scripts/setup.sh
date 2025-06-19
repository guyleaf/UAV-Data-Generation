#!/usr/bin/env bash
set -e

conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate uav_data_generation
blenderproc pip install .

# execute a dummy script to run the blenderproc setup
blenderproc run "$(dirname "$0")/blender/list_gpu_devices_for_cycles.py"
