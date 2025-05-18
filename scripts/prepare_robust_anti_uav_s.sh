#!/usr/bin/env bash
set -eu

root="$HOME/data/Weather"
bash "$(dirname "$0")/datasets/prepare_image2weather.sh" "$root"

out="$HOME/data/UAV/Robust_Anti_UAV_S"
python "$(dirname "$0")/pre_processings/prepare_backgrounds.py" "$root/Image2Weather/dataset" "$out/backgrounds" --dataset-names "Image2Weather" --max-samples 7500
