#!/usr/bin/env bash

#SBATCH --account=project_2002016
#SBATCH --job-name=gec


#SBATCH --partition=gpu

#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:1


module load gcc/8.3.0 cuda/10.1.168

export DIR=/projappl/project_2002016/bert-gec
cd $DIR

conda activate vernet

srun ./train_ru.sh
