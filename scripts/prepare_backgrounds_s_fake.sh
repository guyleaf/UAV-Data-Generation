#!/usr/bin/env bash
set -eu

python scripts/prepare_backgrounds.py ~/data/UAV/Weather_Anti_UAV/originals/Image2Weather ~/data/UAV/Weather_Anti_UAV/originals/FWID ~/data/UAV/Weather_Anti_UAV_S_F --max-samples 3000 --fake-only
