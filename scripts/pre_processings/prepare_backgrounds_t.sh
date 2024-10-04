#!/usr/bin/env bash
set -eu

root="$HOME/data/Weather/FWID_Image2Weather"
python "$(dirname "$0")/prepare_backgrounds.py" "$root/Image2Weather" "$root/FWID" ~/data/UAV/Weather_Anti_UAV_T_NO_RETRY_AREA/backgrounds --max-samples 500
