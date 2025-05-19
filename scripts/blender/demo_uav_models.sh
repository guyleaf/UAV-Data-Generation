#!/usr/bin/env bash
set -eu

# add --temp-dir "$HOME/work/work_dirs/demo_uav_models/tmp" if you encounter OOM

blenderproc run -- "$(dirname "$0")/demo_uav_models.py" \
configs/robust_anti_uav_demo.py \
--out-dir "$HOME/work/work_dirs/demo_uav_models" \
--devices 0 \
--samples 30
