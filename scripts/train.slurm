#!/bin/bash

#SBATCH --job-name=pt3_trian
#SBATCH --partition=gpuA800      
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        
#SBATCH --gres=gpu:1             
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load cuda/12.4.1
conda init bash
source ~/.bashrc
conda activate pointcept
cd /public/home/202420116260/Project/Pointcept
# sh scripts/train.sh -g 1 -d clothes -c reward-standard-ptv3m2 -n reward-standard-ptv3m2
sh scripts/train.sh -g 1 -d clothes -c reward-standard-ptv3m1 -n reward-standard-ptv3m1
# sh scripts/train.sh -g 1 -d clothes -c reward-ptv3m1-mae -n reward-ptv3m1-mae-midium
