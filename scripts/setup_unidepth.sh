#!/usr/bin/env bash
set -e

torch_index="https://download.pytorch.org/whl/cu118"

conda create --name unidepth python=3.10 pip "opencv<5"
eval "$(conda shell.bash hook)"
conda activate unidepth
pip install rich "numpy<2"
pip install -e git+https://github.com/lpiccinelli-eth/UniDepth.git#egg=unidepth --src thirdparty --extra-index-url "$torch_index"
