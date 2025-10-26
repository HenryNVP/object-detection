#!/bin/bash
# Quick start script for DINO-DETR distillation training

set -e  # Exit on error

echo "=========================================="
echo "DINO-DETR Distillation Quick Start"
echo "=========================================="
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Step 2: Download KITTI dataset (optional - can be skipped if already downloaded)
if [ ! -d "kitti_data/training" ]; then
    echo "Step 2: Downloading KITTI dataset..."
    python scripts/download_kitti.py --output-dir ./kitti_data
    echo "✅ Dataset downloaded"
else
    echo "Step 2: KITTI dataset already exists, skipping download"
fi
echo ""

# Step 3: Prepare COCO format dataset
if [ ! -d "kitti_coco" ]; then
    echo "Step 3: Converting KITTI to COCO format..."
    python scripts/prepare_kitti_coco.py \
        --kitti-root ./kitti_data/training \
        --output-dir ./kitti_coco \
        --train-split 0.8 \
        --seed 42
    echo "✅ Dataset converted to COCO format"
else
    echo "Step 3: COCO format dataset already exists, skipping conversion"
fi
echo ""

# Step 4: Train with distillation
echo "Step 4: Training with knowledge distillation..."
echo "This will take a while depending on your hardware..."
python train_distillation.py \
    --data-root ./kitti_coco \
    --teacher-model IDEA-Research/dino-detr-resnet-50 \
    --student-model IDEA-Research/dino-detr-resnet-50 \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --temperature 2.0 \
    --alpha 0.5 \
    --output-dir ./output/distillation \
    --seed 42

echo ""
echo "✅ Training complete!"
echo ""

# Step 5: Evaluate
echo "Step 5: Evaluating the trained model..."
python eval_distillation.py \
    --checkpoint ./output/distillation/best.pth \
    --data-root ./kitti_coco \
    --split val \
    --output ./output/predictions.json

echo ""
echo "=========================================="
echo "🎉 All done!"
echo "=========================================="
echo ""
echo "Your trained model is saved at: ./output/distillation/best.pth"
echo "Predictions are saved at: ./output/predictions.json"
echo ""
echo "To train with different parameters, see README.md for more options."

