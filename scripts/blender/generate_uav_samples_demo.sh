#!/usr/bin/env bash
set -eu

mpiexec -n 2 -x CONSOLE_WIDTH=200 -mca orte_base_help_aggregate 0 -report-bindings \
blenderproc run -- \
"$(dirname "$0")/generate_foregrounds.py" \
configs/robust_anti_uav_demo.py --distributed
