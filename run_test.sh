#!/bin/bash
#SBATCH -J animel2mTest
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=30G
#SBATCH -t 2-00:00:00
#SBATCH -o /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.out
#SBATCH -e /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.err
set -euo pipefail

module load miniconda
source activate /gpfs/milgram/pi/holmes/yz2483/conda_envs/animel2m

cd /gpfs/milgram/home/yz2483/animel2m
pwd

SEED=4710
MODE="anixplore"          # "baseline" or "anixplore"
MODEL_NAME="None"   
FOLDS=("0" "1" "2" "3" "4")
PYTHON="python"          
LEARNER="learner.py"    

for FOLD in "${FOLDS[@]}"; do
    BASE_DIR="out/seed${SEED}_fold${FOLD}"
    if [[ "$MODE" == "baseline" ]]; then
        RUN_NAME="${MODEL_NAME}"
    else
        RUN_NAME="anixplore"
    fi

    CKPT_DIR="${BASE_DIR}/checkpoint/${RUN_NAME}"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "[WARN] Fold ${FOLD}: checkpoint dir not found: ${CKPT_DIR}, skip."
        continue
    fi

    METRICS_CSV="${CKPT_DIR}/metrics.csv"

    # 如果 metrics.csv 不存在，就写 header
    if [[ ! -f "$METRICS_CSV" ]]; then
        echo "ckpt,fold,loss,acc,auc" > "$METRICS_CSV"
    fi

    echo "=== Fold ${FOLD} | MODE=${MODE} | RUN_NAME=${RUN_NAME} ==="
    for CKPT in "${CKPT_DIR}"/epoch=*-val_auc=*.ckpt; do
        # 防止找不到匹配时把字面量写进去
        if [[ ! -f "$CKPT" ]]; then
            echo "[WARN] Fold ${FOLD}: no ckpt found in ${CKPT_DIR}"
            break
        fi

        CKPT_BASENAME=$(basename "$CKPT")

        # 如果已经在 metrics.csv 里面有这一行，就跳过（避免重复测试）
        if grep -q "^${CKPT_BASENAME}," "$METRICS_CSV"; then
            echo "  [SKIP] ${CKPT_BASENAME} already in metrics.csv"
            continue
        fi

        echo "  [TEST] ${CKPT_BASENAME}"

        # 运行 test_only 模式，并抓取 [TEST ] 那一行
        TEST_LINE=$(
            "${PYTHON}" "$LEARNER" \
                --mode "$MODE" \
                --model_name "$MODEL_NAME" \
                --fold "$FOLD" \
                --test_only \
                --ckpt_path "$CKPT" 2>&1 \
            | tee /dev/stderr \
            | grep "\[TEST \]"
        )

        # 解析 Loss / Acc / AUC
        # 格式: [TEST ] Loss: 0.1234 | Acc: 0.5678 | AUC: 0.9876
        LOSS=$(echo "$TEST_LINE" | sed -E 's/.*Loss: ([0-9.]+) \| Acc:.*/\1/')
        ACC=$(echo "$TEST_LINE"  | sed -E 's/.*Acc: ([0-9.]+) \| AUC:.*/\1/')
        AUC=$(echo "$TEST_LINE"  | sed -E 's/.*AUC: ([0-9.]+).*/\1/')

        echo "    Parsed -> loss=${LOSS}, acc=${ACC}, auc=${AUC}"

        # 追加写入 metrics.csv
        echo "${CKPT_BASENAME},${FOLD},${LOSS},${ACC},${AUC}" >> "$METRICS_CSV"
    done
done
