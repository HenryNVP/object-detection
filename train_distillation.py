"""Train DETR with knowledge distillation on KITTI dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.datasets import build_kitti_coco_dataset, collate_fn
from src.distillation import DistillationLoss, DistillationTrainer
from src.models import build_teacher_student_models
from src.utils.device import get_device
from src.utils.seed import seed_all


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DETR with knowledge distillation on KITTI"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-root",
        type=Path,
        default="./kitti_coco",
        help="Root directory of KITTI COCO dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    
    # Model arguments
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="facebook/detr-resnet-50",
        help="Teacher model name from Hugging Face",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="facebook/detr-resnet-50",
        help="Student model name from Hugging Face",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=4,
        help="Number of object classes (including background)",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    
    # Distillation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for distillation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for student loss (1-alpha for distillation loss)",
    )
    
    # Other arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./output/distillation",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu/mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    
    return parser.parse_args()


def build_transforms():
    """Build image transforms for training and validation."""
    # Simple transforms - DINO-DETR handles preprocessing internally
    train_transforms = None  # Let the model's processor handle it
    val_transforms = None
    
    return train_transforms, val_transforms


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    seed_all(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Build datasets
    print("Loading datasets...")
    train_transforms, val_transforms = build_transforms()
    
    train_dataset = build_kitti_coco_dataset(
        split="train",
        data_root=args.data_root,
        transforms=train_transforms,
    )
    
    # Check if validation split exists
    val_dataset = None
    val_loader = None
    try:
        val_dataset = build_kitti_coco_dataset(
            split="val",
            data_root=args.data_root,
            transforms=val_transforms,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        print(f"Loaded {len(val_dataset)} validation samples")
    except ValueError as e:
        print(f"No validation split found: {e}")
        print("Training without validation")
    
    print(f"Loaded {len(train_dataset)} training samples")
    
    # Build data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Build models
    print("Loading models...")
    teacher_model, student_model, image_processor = build_teacher_student_models(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        num_labels=args.num_labels,
        device=device,
    )
    
    print(f"Teacher model: {args.teacher_model}")
    print(f"Student model: {args.student_model}")
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    student_trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,} ({student_trainable:,} trainable)")
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Build distillation loss
    distillation_loss = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
    )
    
    # Build trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        distillation_loss=distillation_loss,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (student weight): {args.alpha}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("-" * 50)
    
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    
    print(f"\nTraining complete! Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

