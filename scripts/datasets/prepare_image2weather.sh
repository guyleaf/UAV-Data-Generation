#!/usr/bin/env bash
set -e

root=$1

if [ -z "$root" ]
then
    echo "Please enter the root folder for storing datasets."
    exit 1
fi

python "$(dirname "$0")/prepare_image2weather_dataset.py" "$root/Image2Weather/original" "$root/Image2Weather/dataset" --all-in-one

python "$(dirname "$0")/clean_classification_datasets.py" "$root/Image2Weather/dataset" --duplicated-dir "$root/Image2Weather/duplicates"
