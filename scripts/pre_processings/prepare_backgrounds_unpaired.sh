#!/usr/bin/env bash
set -eu

root="$HOME/data/Weather/FWID_Image2Weather"
python "$(dirname "$0")/prepare_backgrounds.py" "$root/Image2Weather" "$root/FWID" ~/data/Weather/FWID_Image2Weather_unpaired/original --max-samples 4000 --filling-labels
