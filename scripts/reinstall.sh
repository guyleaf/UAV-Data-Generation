#!/usr/bin/env bash
set -e

yes | blenderproc pip uninstall uav_data_generation
blenderproc pip install ".[blender]"
