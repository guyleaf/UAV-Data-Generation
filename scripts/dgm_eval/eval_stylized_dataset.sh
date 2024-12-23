#!/usr/bin/env bash
set -eu

bg=$1
stylized=$2
out=$3
batchSize="${4:-512}"
gpu="${5:-cuda:0}"

weathers=("foggy" "rainy" "snowy")
bgWeathers=("foggy" "rainy" "snowy")
metrics=("fd" "kd")

# You can set `WEATHERS` environment variable to specify which weathers you want to evaluate.
if [ -v "WEATHERS" ] && [ -n "$WEATHERS" ]; then
    IFS=" " read -ra weathers <<< "$WEATHERS"
    bgWeathers=("${weathers[@]}")
fi

# You can set `BG_WEATHERS` environment variable to specify which weathers in background you want to evaluate.
if [ -v "BG_WEATHERS" ] && [ -n "$BG_WEATHERS" ]; then
    IFS=" " read -ra bgWeathers <<< "$BG_WEATHERS"
fi

if [ "${#bgWeathers[@]}" -ne "${#weathers[@]}" ]; then
    echo "Error: The number of weathers should be the same."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate gen_eval

echo "Output folder: $out"
# rm -rf "$out"

for i in "${!weathers[@]}"
do
    weather=${weathers[i]}
    bgWeather=${bgWeathers[i]}

    python -m dgm_eval "${bg%/}/$bgWeather" "${stylized%/}/$weather" \
        --metrics "${metrics[@]}" \
        --batch_size "$batchSize" --device "$gpu" --model "dinov2" --output_dir "$out/dinov2"

    python -m dgm_eval "${bg%/}/$bgWeather" "${stylized%/}/$weather" \
        --metrics "${metrics[@]}" \
        --batch_size "$batchSize" --device "$gpu" --model "inception" --output_dir "$out/inceptionv3"
done
