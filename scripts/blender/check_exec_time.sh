#!/usr/bin/env bash
set -eu

default_pid=$(ps -o pid -C blender --no-header)

root=$1
pid=${2:-$default_pid}

cmd=$(ps -q "$pid" -o cmd --no-header)
read -ra exec_info <<< "$(ps -q "$pid" -o etime,etimes --no-header)"

echo "Root: $root"
echo "PID: $pid"
echo "CMD: $cmd"
echo

etimes=${exec_info[1]}
num_samples=$(find "$root/foregrounds" -type f | wc -l)
echo "Num of samples: $num_samples"
echo "Exec time: ${exec_info[0]}, $(echo "scale=2; $etimes / $num_samples" | bc)s/sample"
