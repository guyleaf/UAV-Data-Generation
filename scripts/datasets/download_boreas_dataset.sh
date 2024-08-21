#!/usr/bin/env bash
set -e

root=$1

if [ -z "$root" ]
then
    echo "Please enter the target root folder for saving dataset."
    exit 1
fi

# snowing
target="$root/snowing"
mkdir -p "$target"
aws --no-sign-request s3 sync s3://boreas/boreas-2020-12-01-13-26  "$target/boreas-2020-12-01-13-26" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-01-26-10-59  "$target/boreas-2021-01-26-10-59" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-01-26-11-22  "$target/boreas-2021-01-26-11-22" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-11-28-09-18  "$target/boreas-2021-11-28-09-18" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-04-22-15-00  "$target/boreas-2021-04-22-15-00" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"

# rain
target="$root/rain"
mkdir -p "$target"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-07-20-17-33  "$target/boreas-2021-07-20-17-33" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-04-29-15-55  "$target/boreas-2021-04-29-15-55" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-06-29-18-53  "$target/boreas-2021-06-29-18-53" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
aws --no-sign-request s3 sync s3://boreas/boreas-2021-10-26-12-35  "$target/boreas-2021-10-26-12-35" --exclude "*"  --include "camera/*" --include "applanix/*" --include "calib/*"
