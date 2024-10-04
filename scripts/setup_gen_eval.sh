#!/usr/bin/env bash
set -e

conda env create -f "$(dirname "$0")/gen_eval.yml"
