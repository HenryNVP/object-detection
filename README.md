# DINO-DETR Knowledge Distillation for Object Detection

This repository provides a modular implementation for training distilled DINO-DETR models on the KITTI dataset using knowledge distillation.

## Features

- ✅ Modular design with reusable components
- ✅ Knowledge distillation for object detection
- ✅ Support for KITTI dataset in COCO format
- ✅ DINO-DETR model support via Hugging Face Transformers
- ✅ Easy-to-use training and evaluation scripts
- ✅ Configurable via command line or YAML files

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd object-detection

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare KITTI Dataset

First, prepare your KITTI dataset in COCO format. See the `CMPE_KITTI.ipynb` notebook for detailed instructions on:
- Downloading KITTI dataset
- Converting to COCO format
- Creating train/val splits

Expected directory structure:
```
kitti_coco/
├── images/
│   ├── train/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── val/
│       ├── 000050.png
│       └── ...
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

### 2. Train with Knowledge Distillation

```bash
# Basic training with default parameters
python train_distillation.py --data-root ./kitti_coco

# Custom configuration
python train_distillation.py \
    --data-root ./kitti_coco \
    --teacher-model IDEA-Research/dino-detr-resnet-50 \
    --student-model IDEA-Research/dino-detr-resnet-50 \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-4 \
    --temperature 2.0 \
    --alpha 0.5 \
    --output-dir ./output/my_experiment
```

### 3. Evaluate Model

```bash
# Evaluate on validation set
python eval_distillation.py \
    --checkpoint ./output/distillation/best.pth \
    --data-root ./kitti_coco \
    --split val \
    --output ./predictions.json
```

## Project Structure

```
object-detection/
├── src/
│   ├── datasets/          # Dataset loaders
│   │   ├── __init__.py
│   │   └── kitti_coco.py  # KITTI COCO dataset
│   ├── models/            # Model builders
│   │   ├── __init__.py
│   │   └── dino_detr.py   # DINO-DETR models
│   ├── distillation/      # Distillation modules
│   │   ├── __init__.py
│   │   ├── losses.py      # Distillation losses
│   │   └── trainer.py     # Training logic
│   └── utils/             # Utility functions
│       ├── checkpoint.py  # Checkpoint management
│       ├── device.py      # Device selection
│       ├── seed.py        # Random seed
│       └── cli.py         # CLI argument parsing
├── configs/               # Configuration files
│   ├── distillation.yaml  # Distillation config
│   └── README.md
├── train_distillation.py  # Main training script
├── eval_distillation.py   # Evaluation script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-root` | `./kitti_coco` | Root directory of KITTI COCO dataset |
| `--batch-size` | `4` | Batch size for training |
| `--num-workers` | `4` | Number of data loading workers |
| `--teacher-model` | `IDEA-Research/dino-detr-resnet-50` | Teacher model name |
| `--student-model` | `IDEA-Research/dino-detr-resnet-50` | Student model name |
| `--num-labels` | `4` | Number of object classes |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `1e-4` | Weight decay |
| `--temperature` | `2.0` | Temperature for distillation |
| `--alpha` | `0.5` | Weight for student loss |
| `--output-dir` | `./output/distillation` | Output directory |
| `--device` | `auto` | Device (cuda/cpu/mps) |
| `--seed` | `42` | Random seed |

### Distillation Parameters

- **Temperature**: Controls the softness of probability distributions. Higher values (e.g., 4.0) produce softer distributions, which can help the student learn better from the teacher.
  
- **Alpha**: Balances the student loss and distillation loss:
  - `alpha = 1.0`: No distillation (standard training)
  - `alpha = 0.0`: Pure distillation (no ground truth loss)
  - `alpha = 0.5`: Equal weighting (recommended starting point)

## Knowledge Distillation

The implementation uses standard knowledge distillation techniques for object detection:

1. **Teacher Model**: Pre-trained DINO-DETR model (frozen)
2. **Student Model**: Smaller or randomly initialized DINO-DETR model
3. **Loss Function**: 
   ```
   L_total = α * L_student + (1 - α) * L_distillation
   ```
   - L_student: Standard object detection loss (classification + localization)
   - L_distillation: KL divergence between teacher and student logits

### Benefits

- 🎯 Improved accuracy compared to training from scratch
- 📉 Faster convergence
- 🔄 Transfer knowledge from large teacher to smaller student
- 💾 Reduced model size while maintaining performance

## Example Workflow

### Full Training Pipeline

```bash
# 1. Prepare dataset (see CMPE_KITTI.ipynb for details)
# This creates kitti_coco/ directory with proper structure

# 2. Train with distillation
python train_distillation.py \
    --data-root ./kitti_coco \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --temperature 3.0 \
    --alpha 0.5 \
    --output-dir ./experiments/exp1

# 3. Evaluate best model
python eval_distillation.py \
    --checkpoint ./experiments/exp1/best.pth \
    --data-root ./kitti_coco \
    --split val

# 4. Evaluate on different split
python eval_distillation.py \
    --checkpoint ./experiments/exp1/best.pth \
    --data-root ./kitti_coco \
    --split train \
    --output ./train_predictions.json
```

## Model Zoo

You can use different DINO-DETR variants as teacher/student:

| Model | Parameters | Use Case |
|-------|------------|----------|
| `IDEA-Research/dino-detr-resnet-50` | ~41M | Balanced performance |
| `facebook/detr-resnet-50` | ~41M | Alternative backbone |

## Customization

### Using Different Models

```python
# In your code
from src.models import build_teacher_student_models

teacher, student, processor = build_teacher_student_models(
    teacher_model_name="IDEA-Research/dino-detr-resnet-50",
    student_model_name="facebook/detr-resnet-50",
    num_labels=4,
    device="cuda"
)
```

### Custom Dataset

```python
# In your code
from src.datasets import KITTICocoDataset

dataset = KITTICocoDataset(
    root_dir="path/to/images",
    annotation_file="path/to/annotations.json",
    transforms=your_transforms
)
```

### Custom Loss

```python
# In your code
from src.distillation import DistillationLoss

loss_fn = DistillationLoss(
    temperature=4.0,
    alpha=0.3
)
```

## Tips for Best Results

1. **Data Preparation**: Ensure your KITTI dataset is properly converted to COCO format
2. **Temperature**: Start with 2.0-4.0, higher for more knowledge transfer
3. **Alpha**: Start with 0.5, adjust based on validation performance
4. **Learning Rate**: Use 1e-4 for fine-tuning, 5e-5 for more stable training
5. **Batch Size**: Increase if GPU memory allows (improves training stability)

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 2 or 1)
- Reduce image size in dataset preprocessing

### Poor Performance
- Increase `--temperature` (try 3.0-4.0)
- Adjust `--alpha` (try 0.3-0.7)
- Train for more epochs
- Check dataset labels and preprocessing

### Slow Training
- Increase `--num-workers`
- Use mixed precision training (requires code modification)
- Use smaller batch size with gradient accumulation

## References

- [DINO-DETR Paper](https://arxiv.org/abs/2203.03605)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## Citation

If you use this code, please cite the original DINO-DETR paper:

```bibtex
@article{zhang2022dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel and Shum, Heung-Yeung},
  journal={arXiv preprint arXiv:2203.03605},
  year={2022}
}
```

## License

This project is released under the MIT License.

