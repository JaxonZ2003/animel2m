#!/bin/bash
#SBATCH -J animel2m_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
#SBATCH -t 2-00:00:00
#SBATCH -o /logs/animel2m/%x_%j.out
#SBATCH -e /logs/animel2m/%x_%j.err

mkdir -p /logs/animel2m
set -eo pipefail

### EDIT HERE AS NEEDED ###
module load miniconda
source activate /gpfs/milgram/pi/holmes/yz2483/conda_envs/animel2m
cd /gpfs/milgram/home/yz2483/animel2m
pwd

FAKE_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images"
REAL_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img"
SEG_PATH="./segformer_mit-b0.pth"
###########################

MODE="baseline"          # anixplore | baseline
MODEL_NAME="resnet"      # convnext | resnet | vit | frequency | efficientnet | lightweight
FOLD=2

# MODEL_NAME can be pgd if MODE is anixplore

srun python learner.py \
  --mode "$MODE" \
  --model_name "$MODEL_NAME" \
  --fold "$FOLD" \
  --fake_root "$FAKE_ROOT" \
  --real_root "$REAL_ROOT" \
  --seg_path "$SEG_PATH"

echo "[DONE] MODE=$MODE MODEL=$MODEL_NAME FOLD=$FOLD"
