"""Training utilities for knowledge distillation in object detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.checkpoint import save_checkpoint
from .losses import DistillationLoss


class DistillationTrainer:
    """Trainer for knowledge distillation in object detection.
    
    Args:
        teacher_model: Teacher model (frozen)
        student_model: Student model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for student model
        distillation_loss: Distillation loss module
        device: Device to train on
        output_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        distillation_loss: DistillationLoss,
        device: str = "cuda",
        output_dir: Path | str = "./output/distillation",
        image_processor=None,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.distillation_loss = distillation_loss
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_processor = image_processor
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses
        """
        self.student_model.train()
        
        total_loss = 0.0
        total_student_loss = 0.0
        total_distill_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for images, targets in pbar:
            # Process images with DETR image processor if available
            if self.image_processor is not None:
                # Images are PIL Images, process them
                inputs = self.image_processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
            else:
                # Images are already tensors
                pixel_values = torch.stack([img.to(self.device) for img in images])
            
            # Move targets to device and add orig_size for DETR
            processed_targets = []
            for i, target in enumerate(targets):
                t = {k: v.to(self.device) for k, v in target.items()}
                # Add orig_size if not present (needed for DETR)
                if "orig_size" not in t and self.image_processor is not None:
                    # Get original image size from PIL image
                    if hasattr(images[i], 'size'):  # PIL Image
                        w, h = images[i].size
                        t["orig_size"] = torch.tensor([h, w]).to(self.device)
                processed_targets.append(t)
            
            # Forward pass - teacher
            with torch.no_grad():
                teacher_outputs = self.teacher_model(pixel_values=pixel_values)
            
            # Forward pass - student
            student_outputs = self.student_model(pixel_values=pixel_values, labels=processed_targets)
            
            # Compute distillation loss
            loss_dict = self.distillation_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                targets=processed_targets,
            )
            
            loss = loss_dict["loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_student_loss += loss_dict.get("student_loss", torch.tensor(0.0)).item()
            total_distill_loss += loss_dict.get("distillation_loss", torch.tensor(0.0)).item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "student": f"{loss_dict.get('student_loss', torch.tensor(0.0)).item():.4f}",
                "distill": f"{loss_dict.get('distillation_loss', torch.tensor(0.0)).item():.4f}",
            })
        
        return {
            "train_loss": total_loss / num_batches,
            "train_student_loss": total_student_loss / num_batches,
            "train_distill_loss": total_distill_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the student model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.student_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for images, targets in pbar:
            # Process images with DETR image processor if available
            if self.image_processor is not None:
                # Images are PIL Images, process them
                inputs = self.image_processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
            else:
                # Images are already tensors
                pixel_values = torch.stack([img.to(self.device) for img in images])
            
            # Move targets to device and add orig_size for DETR
            processed_targets = []
            for i, target in enumerate(targets):
                t = {k: v.to(self.device) for k, v in target.items()}
                # Add orig_size if not present (needed for DETR)
                if "orig_size" not in t and self.image_processor is not None:
                    # Get original image size from PIL image
                    if hasattr(images[i], 'size'):  # PIL Image
                        w, h = images[i].size
                        t["orig_size"] = torch.tensor([h, w]).to(self.device)
                processed_targets.append(t)
            
            # Forward pass
            outputs = self.student_model(pixel_values=pixel_values, labels=processed_targets)
            loss = outputs.get("loss", torch.tensor(0.0))
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        return {
            "val_loss": total_loss / num_batches,
        }
    
    def train(self, num_epochs: int, save_every: int = 1) -> None:
        """Train the student model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        best_val_loss = float("inf")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  - Student Loss: {train_metrics['train_student_loss']:.4f}")
            print(f"  - Distillation Loss: {train_metrics['train_distill_loss']:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Save best model
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint(epoch, best_val_loss, "best")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, best_val_loss, f"epoch_{epoch}")
        
        print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch: int, val_loss: float, name: str = "checkpoint") -> None:
        """Save a checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            name: Checkpoint name
        """
        checkpoint_path = self.output_dir / f"{name}.pth"
        save_checkpoint(
            model=self.student_model,
            optimizer=self.optimizer,
            epoch=epoch,
            best_acc=val_loss,  # Using val_loss instead of acc
            path=checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

