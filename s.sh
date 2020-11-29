#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=2-23:50
#SBATCH --output=%N-%j.out

module load python/3.8 nixpkgs/16.09 gcc/7.3.0 cuda/10.2

source ~/finetune1/bin/activate

wandb login

cd ~/projects/def-gbader/qiwu/CellTyper

python app.py --json_args config.json