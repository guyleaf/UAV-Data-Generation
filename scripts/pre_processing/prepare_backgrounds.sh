#!/usr/bin/env bash
set -eu

python "$(dirname "$0")/prepare_backgrounds.py" ~/data/UAV/Weather_Anti_UAV/originals/Image2Weather ~/data/UAV/Weather_Anti_UAV/originals/FWID ~/data/UAV/Weather_Anti_UAV --max-samples 7000
