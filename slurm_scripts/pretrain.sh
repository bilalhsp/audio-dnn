#!/bin/sh

#SBATCH	-A training
#standby
#jgmakin-n
#training
#debug
# --constraint=F|G|I|K|D|B|H|J|C|N 

# High Mem GPUs: F|G|I|K|D
# very Fast GPUs: F|K
# Fast GPUs: B|D
# Slow GPUs: E

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --exclusive

#--mem=0 is no longer supported, 
# If you want to use the entire node's memory,
# you can submit the job with the --exclusive option
# otherwise specify memory explicitly e.g. --mem=20G

hostname
NUMBA_DISABLE_INTEL_SVML=1
echo "Total tasks: $SLURM_NTASKS"
echo "Total tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Number of nodes: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"


module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5
# module load conda-env/huggingface-py3.8.5
# module load gcc/9.3.0
#module load conda-env/wav2letter_pretrained-py3.8.5
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker,eth,lo

torchrun --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --node_rank=$SLURM_NODEID \
    ../scripts/pretrain_wav2vec2.py "$@" -t $SLURM_NTASKS -n $SLURM_CPUS_PER_TASK




