#!/usr/bin/env bash
#SBATCH --job-name=generate_uav_samples_job         # 工作名稱
#SBATCH --partition=normal                          # 使用的 partition (請根據你的系統修改)
#SBATCH --time=48:00:00                             # 執行時間上限 (小時:分鐘:秒)
#SBATCH --account=MST114107

#SBATCH --nodes=1                                   # (-N) Maximum number of nodes to be allocated
#SBATCH --ntasks=8                                  # (-n) Number of tasks to run
#SBATCH --ntasks-per-node=8                         # Maximum number of tasks per node
#SBATCH --cpus-per-task=12                          # (-c) Number of cores per task

#SBATCH --gpus=8                                    # (-G) Number of GPUs to run
#SBATCH --gpus-per-node=8                           # Number of GPUs per node

#SBATCH -o %x_%j_node_%n.log                        # output file (%j expands to jobId)
#SBATCH -e %x_%j_node_%n.err.log                    # output file (%j expands to jobId)
#SBATCH --mail-type=END,FAIL                        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=leaf.ying.cs11@nycu.edu.tw      # Where to send mail.  Set this to your email address

set -eu

name=$1

# logging in verbose mode (show all infos in every rank)
export LOGGING_VERBOSE=1
# let the logging know the console width
export CONSOLE_WIDTH=${CONSOLE_WIDTH:-200}
export DISTRIBUTED=1

# WORKAROUND: UCX 1.16 will fail if there is a NIC having empty IP address.
export UCX_PROTO_ENABLE="n"
# export UCX_LOG_LEVEL="info"
# export UCX_PROTO_INFO="y"
# export UCX_TLS="^tcp"
# export UCX_NET_DEVICES="mlx5_3:1,mlx5_4:1,mlx5_5:1"

module purge
module load openmpi
module list

# srun --mpi=pmix python "scripts/blender/show_mpi_rank.py"
srun --mpi=pmix "scripts/blender/generate_uav_samples_$name.sh"
# srun blenderproc run "scripts/blender/list_gpu_devices_for_cycles.py"
