#!/usr/bin/env bash
set -eu

blenderproc run -- "$(dirname "$0")/demo_randomization.py" \
~/data/UAV/blender/UAVs.blend \
~/data/UAV/blender/assets/backgrounds/studiolights/city.exr \
--out-dir ~/work/work_dirs/demo_randomization_adaptive_1_0 \
--devices 0 \
--x-range -30 30 --y-range -30 30 --samples 10 \
--render-max-samples 1024 --render-resolution 1920 1920
