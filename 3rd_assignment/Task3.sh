#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home2/hmnshpl/projects/results/ANLP-A3-Task3.txt
#SBATCH --nodelist gnode092
#SBATCH --time=96:00:00
#SBATCH --mail-user=himanshu.pal@research.iiit.ac.in
#SBATCH --mail-type=ALL


source ~/.bashrc

conda activate anlp

python Task3.py


