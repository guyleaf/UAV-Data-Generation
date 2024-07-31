#!/usr/bin/env bash
set -eu

blenderproc run -- blender/main.py ~/data/UAV/blender/UAVs.blend ~/data/UAV/blender/assets/backgrounds/studiolights/city.exr ~/data/Weather/Weather-Anti-UAV/backgrounds --out-dir ~/data/Weather/Weather-Anti-UAV --devices 0 1 2 3 4 5 6 7 --x-range -30 30 --y-range -30 30 --scale-range 0.01 0.3 --max-iou 0 --render-resolution 2560 2560 --render-tile-size 8192 --max-samples 10
