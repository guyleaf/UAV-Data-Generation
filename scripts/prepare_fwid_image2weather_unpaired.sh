#!/usr/bin/env bash
set -eu

root="$HOME/data/Weather"
out="$root/FWID_Image2Weather"
bash "$(dirname "$0")/datasets/prepare_fwid_image2weather.sh" "$root" "$out"

root="$out"
out="$root/FWID_Image2Weather_unpaired"
python "$(dirname "$0")/pre_processings/prepare_backgrounds.py" "$root/Image2Weather" "$root/FWID" "$out/original" --max-samples 4000 --filling-labels

root="$out"
python "$(dirname "$0")/datasets/combine_classification_datasets.py" "$root/original" "$root/combined" --all-in-one
# train = 3200, val = 800
python "$(dirname "$0")/datasets/split_classification_dataset.py" "$root/combined" "$root/dataset" --val-ratio 0.2
