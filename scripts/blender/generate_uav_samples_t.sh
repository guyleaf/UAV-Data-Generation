#!/usr/bin/env bash
set -eu

# blenderproc run -- "$(dirname "$0")/generate_foregrounds.py" \
# ~/data/UAV/blender/UAVs.blend \
# ~/data/UAV/blender/assets/backgrounds/studiolights/city.exr \
# ~/data/UAV/Weather_Anti_UAV_T/backgrounds \
# --out-dir ~/data/UAV/Weather_Anti_UAV_T \
# --devices 4 \
# --x-range -30 30 --y-range -30 30 --scale-range 0.002 0.03 --max-iof 0.2 --max-samples 6 \
# --render-max-samples 1024 --render-resolution 1920 1920 --render-tile-size 1920

# blenderproc run -- "$(dirname "$0")/generate_foregrounds.py" \
# ~/data/UAV/blender/UAVs.blend \
# ~/data/UAV/blender/assets/backgrounds/studiolights/city.exr \
# ~/data/UAV/Weather_Anti_UAV_T_NO_RETRY/backgrounds \
# --out-dir ~/data/UAV/Weather_Anti_UAV_T_NO_RETRY \
# --devices 0 \
# --x-range -30 30 --y-range -30 30 --scale-range 0.002 0.2 --max-iof 0.2 --max-samples 6 \
# --render-max-samples 256 --render-resolution 1920 1920 --render-tile-size 1920 --resume

# blenderproc run -- "$(dirname "$0")/generate_foregrounds.py" \
# ~/data/UAV/blender/UAVs.blend \
# ~/data/UAV/blender/assets/backgrounds/studiolights/city.exr \
# ~/data/UAV/Weather_Anti_UAV_T_NO_RETRY_2e-3_3e-2/backgrounds \
# --out-dir ~/data/UAV/Weather_Anti_UAV_T_NO_RETRY_2e-3_3e-2 \
# --devices 0 \
# --x-range -30 30 --y-range -30 30 --scale-range 0.002 0.03 --max-iof 0.2 --max-samples 6 \
# --render-max-samples 128 --render-resolution 1920 1920 --render-tile-size 1920

blenderproc run -- "$(dirname "$0")/generate_foregrounds.py" \
configs/weather_anti_uav_t.py \
--devices 0 --resume
