#!/usr/bin/env bash
set -e

python "$(dirname "$0")/combine_classification_datasets.py" ~/data/Weather/FWID_Image2Weather_unpaired/original ~/data/Weather/FWID_Image2Weather_unpaired/combined --all-in-one

python "$(dirname "$0")/split_classification_dataset.py" ~/data/Weather/FWID_Image2Weather_unpaired/combined ~/data/Weather/FWID_Image2Weather_unpaired/dataset --val-ratio 0.2
