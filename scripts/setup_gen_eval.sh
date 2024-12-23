#!/usr/bin/env bash
set -e

torch_index="https://download.pytorch.org/whl/cu118"

conda create --name gen_eval python=3.10 pip
eval "$(conda shell.bash hook)"
conda activate gen_eval
pip install -e git+https://github.com/layer6ai-labs/dgm-eval.git#egg=dgm_eval --src thirdparty --extra-index-url "$torch_index"
pip install torch-fidelity pytorch-fid --extra-index-url "$torch_index"
