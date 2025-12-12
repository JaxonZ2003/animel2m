#!/bin/bash
#SBATCH -J animel2mInterpretation
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00
#SBATCH -o /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.out
#SBATCH -e /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.err

set -euo pipefail

module load miniconda
source activate /gpfs/milgram/pi/holmes/yz2483/conda_envs/animel2m

cd /gpfs/milgram/home/yz2483/animel2m
pwd

python Interpretation.py