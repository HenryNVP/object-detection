"""Distillation losses for object detection models."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for object detection.
    
    Combines standard detection loss with distillation loss from teacher model.
    
    Args:
        temperature: Temperature for softening logits
        alpha: Weight for balancing student loss and distillation loss
               loss = alpha * student_loss + (1 - alpha) * distillation_loss
    """
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss.
        
        Args:
            student_outputs: Dictionary containing student model outputs with 'loss' and 'logits'
            teacher_outputs: Dictionary containing teacher model outputs with 'logits'
            targets: Ground truth targets
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Get student loss (standard detection loss)
        student_loss = student_outputs.get("loss", torch.tensor(0.0, device=student_outputs["logits"].device))
        
        # Extract logits
        student_logits = student_outputs["logits"]  # [batch, num_queries, num_classes]
        teacher_logits = teacher_outputs["logits"]  # [batch, num_queries, num_classes]
        
        # Compute distillation loss (KL divergence)
        # Apply softmax with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        return {
            "loss": total_loss,
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
        }


class FeatureDistillationLoss(nn.Module):
    """Feature-based distillation loss for intermediate representations.
    
    Args:
        lambda_feat: Weight for feature distillation loss
    """
    
    def __init__(self, lambda_feat: float = 0.1):
        super().__init__()
        self.lambda_feat = lambda_feat
        
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature distillation loss (MSE).
        
        Args:
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features
            
        Returns:
            Feature distillation loss
        """
        # Ensure same dimensions
        if student_features.shape != teacher_features.shape:
            # Apply adaptive pooling or projection if needed
            student_features = F.adaptive_avg_pool2d(student_features, teacher_features.shape[-2:])
        
        loss = F.mse_loss(student_features, teacher_features)
        return self.lambda_feat * loss

