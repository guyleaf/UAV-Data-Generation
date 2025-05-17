#!/usr/bin/env bash
set -eu

# add --temp-dir "$HOME/work/work_dirs/demo_randomization/tmp" if you encounter OOM

blenderproc run -- "$(dirname "$0")/demo_randomization.py" \
configs/weather_anti_uav_demo.py \
--out-dir "$HOME/work/work_dirs/demo_randomization" \
--devices 0 \
--samples 10
