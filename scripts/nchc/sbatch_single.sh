#!/usr/bin/env bash
#SBATCH --job-name=single_node_job      # 工作名稱
#SBATCH --partition=dev                 # 使用的 partition (請根據你的系統修改)
#SBATCH --time=00:30:00                 # 執行時間上限 (小時:分鐘:秒)
#SBATCH --account=MST114107

#SBATCH --nodes=1                                   # (-N) Maximum number of nodes to be allocated
#SBATCH --ntasks=1                                  # (-n) Number of tasks to run
#SBATCH --ntasks-per-node=1                         # Maximum number of tasks per node
#SBATCH --cpus-per-task=8                           # (-c) Number of cores per task

#SBATCH --mem=128G                                  # RAM per node

#SBATCH --gpus=1                                    # (-G) Number of GPUs to run
#SBATCH --gpus-per-node=1                           # Number of GPUs per node

#SBATCH -o %x_%j.log                                # output file (%j expands to jobId)
#SBATCH -e %x_%j.err.log                            # output file (%j expands to jobId)
#SBATCH --mail-type=END,FAIL                        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=leaf.ying.cs11@nycu.edu.tw      # Where to send mail.  Set this to your email address

set -eu

program=$1

# let the logging know the console width
export CONSOLE_WIDTH=${CONSOLE_WIDTH:-200}

module purge
module load openmpi
module list

srun --mpi=pmix "$program" "${@:2}"
