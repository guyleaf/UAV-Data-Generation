#!/usr/bin/env bash
set -e

python3 -m venv .venv
# shellcheck source=/dev/null
source ".venv/bin/activate"

pip install -U pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e .[dev]
blenderproc pip install .
