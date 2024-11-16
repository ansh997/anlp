#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=./part2_1.txt
#SBATCH --nodelist gnode080
#SBATCH --time=96:00:00

source ~/.bashrc

conda activate anlp

python part2_1.py