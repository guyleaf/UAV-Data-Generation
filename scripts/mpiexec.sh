#!/usr/bin/env bash
set -eu

script=$1
num_procs=${2:-2}

console_width=${CONSOLE_WIDTH:-200}

mpiexec -n "$num_procs" -x CONSOLE_WIDTH="$console_width" -x DISTRIBUTED=1 -mca orte_base_help_aggregate 0 -report-bindings \
    "$script" "${@:2}"
