#!/usr/bin/env bash
set -eu

root="$HOME/data/Weather"
out="$root/FWID_Image2Weather"
bash "$(dirname "$0")/datasets/prepare_fwid_image2weather.sh" "$root" "$out"

root="$out"
python "$(dirname "$0")/pre_processings/prepare_backgrounds.py" "$root/Image2Weather" "$root/FWID" ~/data/UAV/Weather_Anti_UAV_S/backgrounds --max-samples 3000
