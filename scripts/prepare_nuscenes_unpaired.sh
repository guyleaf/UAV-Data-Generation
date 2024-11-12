#!/usr/bin/env bash
set -eu

root="$HOME/data/Weather/nuScenes"
out="$HOME/data/Weather/nuScenes_unpaired_new"
# train = 3200, val = 800
python "$(dirname "$0")/datasets/prepare_nuscenes_dataset.py" "$root" "$out" --max-samples 4000 --val-ratio 0.2

# root="$out"
# out="$root/nuScenes_unpaired"
# python "$(dirname "$0")/pre_processings/prepare_backgrounds.py" "$root" "$out" --max-samples 4000 --filling-labels
