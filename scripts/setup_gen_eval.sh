#!/usr/bin/env bash
set -e

# conda env create -f "$(dirname "$0")/gen_eval.yml"
conda create --name gen_eval python=3.10 pip
eval "$(conda shell.bash hook)"
conda activate gen_eval
pip install -e git+https://github.com/layer6ai-labs/dgm-eval.git#egg=dgm_eval --src thirdparty --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-fidelity
