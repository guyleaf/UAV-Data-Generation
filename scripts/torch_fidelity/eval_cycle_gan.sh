#!/usr/bin/env bash
set -eu

real=$1
exp=$2
batchSize="${3:-512}"
gpu="${4:-cuda:0}"
fake_regex="${5:-fake_*}"

eval "$(conda shell.bash hook)"
conda activate gen_eval


out="${exp%/}/torch_fidelity"
echo "Output folder: $out"

rm -rf "$out" && mkdir -p "$out"

echo "Inceptionv3 results" | tee -a "$out/output.log"
python "$(dirname "$0")/find_best_score_cycle_gan.py" "$real" "$exp" "$out/inceptionv3" --fake-dir "$fake_regex" --batch-size "$batchSize" --device "$gpu" | tee -a "$out/output.log"
