#!/bin/bash
#SBATCH -J anixplore_exp
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.out
#SBATCH -e /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.err

set -eo pipefail

# === 1. 环境设置 (根据你的 Traceback 修改) ===
module load miniconda
# 激活你之前的报错信息中显示的 environment
source activate /gpfs/milgram/pi/holmes/yz2483/conda_envs/animel2m

# 切换到项目目录
cd /gpfs/milgram/home/yz2483/animel2m
pwd

# 确保日志文件夹存在
mkdir -p out/checkpoint
mkdir -p out/logs

# === 2. 全局变量设置 (请确认路径) ===
FAKE_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images"        # 修改为你的实际 fake_data 路径
REAL_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img"        # 修改为你的实际 real_data 路径
SEG_PATH="./segformer_mit-b0.pth"        # 修改为你的 SegFormer 权重路径
BATCH_SIZE=16
EPOCHS=20

# === 3. 任务分配逻辑 ===
# Array ID: 0 -> AniXplore
# Array ID: 1 -> Baseline: ConvNeXt
# Array ID: 2 -> Baseline: ResNet
# Array ID: 3 -> Baseline: ViT

AID=${SLURM_ARRAY_TASK_ID:-0}

if [ $AID -eq 0 ]; then
    MODE="anixplore"
    MODEL_NAME="none" # AniXplore 不需要 model_name，占位用
    echo "[Task $AID] Running Model: AniXplore"

elif [ $AID -eq 1 ]; then
    MODE="baseline"
    MODEL_NAME="convnext"
    echo "[Task $AID] Running Baseline: ConvNeXt"

elif [ $AID -eq 2 ]; then
    MODE="baseline"
    MODEL_NAME="frequency"
    echo "[Task $AID] Running Baseline: FrequencyAwareBaseline"

elif [ $AID -eq 3 ]; then
    MODE="baseline"
    MODEL_NAME="efficientnet"
    echo "[Task $AID] Running Baseline: EfficientNetBaseline"
else
    echo "Error: Array ID $AID out of range (0-3)"
    exit 1
fi

# === 4. 执行命令 ===
# 假设你的 python 脚本叫 learner.py (根据之前的 traceback)
# 如果叫 train.py 请修改文件名
srun python learner.py \
  --mode "$MODE" \
  --model_name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --fake_root "$FAKE_ROOT" \
  --real_root "$REAL_ROOT" \
  --seg_path "$SEG_PATH"

echo "[Task $AID] Finished."