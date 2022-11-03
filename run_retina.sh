#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx
##
#SBATCH --job-name=ship_retina
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##
#SBATCH --gres=gpu:hgx:1

hostname
date

module add CUDA/11.2.2
module add ANACONDA/2020.11

python /home1/wonhyung64/Github/retinanet/run.py --data-dir /home1/wonhyung64/data  --name ship