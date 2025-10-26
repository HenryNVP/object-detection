# Configuration Files

This directory contains configuration files for training distilled DINO-DETR models.

## Files

- `distillation.yaml`: Main configuration for knowledge distillation training

## Usage

You can override any configuration parameter via command line arguments:

```bash
# Use default configuration
python train_distillation.py

# Override specific parameters
python train_distillation.py --epochs 20 --batch-size 8 --lr 5e-5

# Specify different teacher/student models
python train_distillation.py \
    --teacher-model IDEA-Research/dino-detr-resnet-50 \
    --student-model IDEA-Research/dino-detr-resnet-50 \
    --temperature 3.0 \
    --alpha 0.7
```

## Parameters

### Data Parameters
- `data_root`: Root directory of KITTI COCO dataset
- `num_labels`: Number of object classes (including background)
- `batch_size`: Batch size for training
- `num_workers`: Number of data loading workers

### Model Parameters
- `teacher_model`: Teacher model name from Hugging Face
- `student_model`: Student model name from Hugging Face

### Training Parameters
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `weight_decay`: Weight decay for optimizer
- `save_every`: Save checkpoint every N epochs

### Distillation Parameters
- `temperature`: Temperature for softening logits (higher = softer)
- `alpha`: Weight for student loss (0.0 = pure distillation, 1.0 = no distillation)

### Other Parameters
- `output_dir`: Output directory for checkpoints
- `seed`: Random seed
- `device`: Device to use (cuda/cpu/mps)

