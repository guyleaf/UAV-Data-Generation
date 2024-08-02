#!/usr/bin/env bash
set -eu

blenderproc run -- blender/main.py \
~/data/UAV/blender/UAVs.blend \
~/data/UAV/blender/assets/backgrounds/studiolights/city.exr \
~/data/Weather/Weather_Anti_UAV_S/backgrounds \
--out-dir ~/data/Weather/Weather_Anti_UAV_S \
--devices 4 \
--x-range -30 30 --y-range -30 30 --scale-range 0.01 0.4 --max-iof 0.2 --max-samples 6 \
--render-max-samples 1024 --render-resolution 2560 2560 --render-tile-size 2560
