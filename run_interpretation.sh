#!/bin/bash
"""
Runner script for model interpretation
This script evaluates all trained models found in the checkpoint directory
"""

# Default paths - adjust these according to your setup
FAKE_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images"
REAL_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img"
CHECKPOINT_DIR="out/checkpoint"
SAVE_DIR="interpretations"

# Function to run interpretation for a specific model
interpret_model() {
    model=$1
    echo "========================================"
    echo "Interpreting model: $model"
    echo "========================================"
    
    python interpret_models.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --models "$model" \
        --fake_root "$FAKE_ROOT" \
        --real_root "$REAL_ROOT" \
        --n_samples 8 \
        --save_dir "$SAVE_DIR" \
        --img_size 224 \
        --batch_size 1
}

# Check if specific model is requested
if [ "$1" != "" ]; then
    interpret_model "$1"
else
    echo "Running interpretation for all available models..."
    
    # Run for all baseline models
    for model in convnext resnet vit frequency efficientnet dualstream lightweight anixplore; do
        if [ -d "$CHECKPOINT_DIR/$model" ]; then
            interpret_model "$model"
        else
            echo "Skipping $model - no checkpoint found"
        fi
    done
    
    echo ""
    echo "========================================"
    echo "Creating combined comparison visualization..."
    echo "========================================"
    
    # Run once more to create combined visualization with all models
    python interpret_models.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --fake_root "$FAKE_ROOT" \
        --real_root "$REAL_ROOT" \
        --n_samples 8 \
        --save_dir "$SAVE_DIR" \
        --img_size 224
fi

echo "Interpretation complete! Check $SAVE_DIR/ for results."