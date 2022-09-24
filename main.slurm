#!/bin/bash

#SBATCH --partition=gpu_7d1g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=predict_mhc.out
#SBATCH --error=predict_mhc.err

source /home/yipingzou2/.bashrc
conda activate env1
cd /home/yipingzou2/predict-mhc

python3 main.py
