#!/usr/bin/env bash
set -eu

out="$HOME/data/Weather/boreas_unpaired"
# train = 3200, val = 800
python "$(dirname "$0")/datasets/prepare_boreas_dataset.py" "$out" --max-samples 4000 --val-ratio 0.2 --max-workers 8
