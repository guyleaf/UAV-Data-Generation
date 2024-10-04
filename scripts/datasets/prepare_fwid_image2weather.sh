#!/usr/bin/env bash
set -e

root=$1

if [ -z "$root" ]
then
    echo "Please enter the root folder for storing datasets."
    exit 1
fi

python "$(dirname "$0")/prepare_image2weather_dataset.py" "$root/Image2Weather/original" "$root/Image2Weather/dataset" --all-in-one
python "$(dirname "$0")/prepare_fwid_dataset.py" "$root/FWID/original" "$root/FWID/dataset" --all-in-one

python "$(dirname "$0")/clean_classification_datasets.py" "$root/Image2Weather/dataset" --duplicated-dir "$root/Image2Weather/duplicates"
python "$(dirname "$0")/clean_classification_datasets.py" "$root/FWID/dataset" --duplicated-dir "$root/FWID/duplicates"

out="$root/FWID_Image2Weather"
mkdir -p "$out"
cp -r "$root/Image2Weather/dataset" "$out/Image2Weather"
cp -r "$root/FWID/dataset" "$out/FWID"
python "$(dirname "$0")scripts/datasets/clean_classification_datasets.py" "$out/Image2Weather" "$out/FWID" --duplicated-dir "$out/duplicates"

# for weather stylization evaluation
python "$(dirname "$0")/combine_classification_datasets.py" "$out/Image2Weather" "$out/FWID" "$out/combined" --all-in-one
