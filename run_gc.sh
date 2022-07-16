#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx
##
#SBATCH --job-name=frcnn_voc07
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##
#SBATCH --gres=gpu:hgx:1

hostname
date

module add CUDA/11.2.2
module add ANACONDA/2020.11

python /home1/wonhyung64/Github/ship_detection/main.py --data-dir /home1/wonhyung64/data --name gc --anchor-scales 8. 16. 32. --pos-threshold 0.6
python /home1/wonhyung64/Github/ship_detection/main.py --data-dir /home1/wonhyung64/data --name gc --anchor-scales 8. 16. 32. --pos-threshold 0.55
python /home1/wonhyung64/Github/ship_detection/main.py --data-dir /home1/wonhyung64/data --name gc --anchor-scales 8. 16. 32. --pos-threshold 0.5
