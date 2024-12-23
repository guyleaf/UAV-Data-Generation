#!/usr/bin/env bash
set -eu

blenderproc run -- "$(dirname "$0")/generate_foregrounds.py" \
configs/weather_anti_uav_s.py \
--devices 4
