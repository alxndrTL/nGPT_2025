#!/bin/bash
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4         # number of gpus per node
#SBATCH --time=3:00:00              # time limits: here 1 hour
#SBATCH --error=logs/experiment3.err            # standard error file
#SBATCH --output=logs/experiment3.out           # standard output file
#SBATCH --account=BOOST_LCustodi       # account name
#SBATCH --partition=boost_usr_prod # partition name for prod

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline

# Usage: bash full_config.sh [RUN_NAME] [NUM_GPUS=1]
RUN_NAME=${1}
NUM_GPUS=${2:-1}

# WARNING!! MUON+DRAGON = bad perfs

torchrun --nproc_per_node=$NUM_GPUS train_ngpt.py
