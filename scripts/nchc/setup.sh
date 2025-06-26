#!/usr/bin/env bash
set -eu

# load openmpi for mpi4py
module purge
module load openmpi

"$(dirname "$0")/../setup.sh"
