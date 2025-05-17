#!/usr/bin/env bash
set -eu

root=$1
gpu=${2:-"cuda:0"}

# https://www.emacswiki.org/emacs/RegularExpression
mapfile -t datasets < <(find "$root/harmonized" -maxdepth 1 ! -path "$root/harmonized" -type d)
echo "Found ${#datasets[@]} image folders."

eval "$(conda shell.bash hook)"
conda activate unidepth

for dataset in "${datasets[@]}"
do
    name=$(basename "$dataset")
    out="$root/depths_unidepthv2/$name"
    python "$(dirname "$0")/unidepth/estimate_depth.py" "$dataset" "$out" --numpy --demo --no-refine --device "$gpu"

    out="$root/depths_unidepthv2/${name}_refined_fastbilateral"
    python "$(dirname "$0")/unidepth/estimate_depth.py" "$dataset" "$out" --numpy --demo --rfilter fastbilateral --device "$gpu"

    out="$root/depths_unidepthv2/${name}_refined_guided"
    python "$(dirname "$0")/unidepth/estimate_depth.py" "$dataset" "$out" --numpy --demo --rfilter guided --device "$gpu"
done
