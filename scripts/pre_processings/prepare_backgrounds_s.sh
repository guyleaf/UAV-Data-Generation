#!/usr/bin/env bash
set -eu

python "$(dirname "$0")/prepare_backgrounds.py" ~/data/Weather/FWID_Image2Weather/Image2Weather ~/data/Weather/FWID_Image2Weather/FWID ~/data/UAV/Weather_Anti_UAV_S --max-samples 3000
