#!/usr/bin/env bash
set -eu

real=$1
exp=$2
batchSize="${3:-512}"
gpu="${4:-cuda:0}"
fake_regex="${5:-fake_B}"

eval "$(conda shell.bash hook)"
conda activate gen_eval

metrics=("fd" "kd")

# find test_ prefix folders
mapfile -t fakes < <(find "$exp" -type d -regex ".*/test_[0-9]+/images/$fake_regex$")
if [ ${#fakes[@]} -eq 0 ]; then
    echo "Cannot found any results in $exp."
    exit 1
fi

# fakes=("${fakes[@]/%/'/images/fake_B'}")
out="${exp%/}/dgm_eval"

echo "Found ${#fakes[@]} results in $exp!"
echo "Output folder: $out"

# rm -rf "$out"

python -m dgm_eval "$real" "${fakes[@]}" \
    --metrics "${metrics[@]}" \
    --batch_size "$batchSize" --device "$gpu" --model "dinov2" --output_dir "$out/dinov2"

python -m dgm_eval "$real" "${fakes[@]}" \
    --metrics "${metrics[@]}" \
    --batch_size "$batchSize" --device "$gpu" --model "inception" --output_dir "$out/inceptionv3"

echo "DINOv2 results" | tee -a "$out/output.log"
python "$(dirname "$0")/find_best_score.py" "$out/dinov2" 2>&1 | tee -a "$out/output.log"
echo "" | tee -a "$out/output.log"
echo "Inceptionv3 results" | tee -a "$out/output.log"
python "$(dirname "$0")/find_best_score.py" "$out/inceptionv3" 2>&1 | tee -a "$out/output.log"
