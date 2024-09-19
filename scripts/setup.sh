#!/usr/bin/env bash
set -e

conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate uav_data_generation
blenderproc pip install ".[blender]"
