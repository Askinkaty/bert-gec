#!/usr/bin/env bash

#SBATCH --account=project_2002016
#SBATCH --job-name=gec


#SBATCH --partition=gpu

#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:4


module load gcc/8.3.0 cuda/10.1.168

export DIR=/projappl/project_2002016/bert-gec/scripts
cd $DIR

#conda activate bert-gec
conda activate gpt

#srun ./preprocess.sh
srun ./pretrain_ru.sh
#srun ./train_ru.sh
#srun ./generate_ru.sh /scratch/project_2002016/datasets/data-gec/test.src gpu
