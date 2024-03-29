#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1
##
#SBATCH --job-name=ship_frcnn
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##
#SBATCH --gres=gpu:rtx3090:1

hostname
date

module add CUDA/11.2.2
module add ANACONDA/2020.11

python /home1/wonhyung64/Github/ship_detection/run.py --name ship --data-dir /home1/wonhyung64/data --anchor-scales 32. 64. 128.
