#!/usr/bin/env bash
set -e

python3 -m venv .venv
# shellcheck source=/dev/null
source ".venv/bin/activate"

pip install -U pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://download.blender.org/pypi/ -e .[dev]
blenderproc pip install .

# execute a dummy script to run the blenderproc setup
blenderproc run "$(dirname "$0")/blender/list_gpu_devices_for_cycles.py"
