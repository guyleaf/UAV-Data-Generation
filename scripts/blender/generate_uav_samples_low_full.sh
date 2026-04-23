#!/usr/bin/env bash
set -eu

blenderproc run -- \
    "$(dirname "$0")/generate_foregrounds.py" \
    configs/robust_anti_uav_low_full.py --resume
