#!/usr/bin/env bash
set -eu

python "$(dirname "$0")/prepare_backgrounds.py" ~/data/Weather/FWID_Image2Weather/Image2Weather ~/data/Weather/FWID_Image2Weather/FWID ~/data/Weather/FWID_Image2Weather_unpaired/original --max-samples 4000 --filling-labels
