"""DINO-DETR model builders for object detection."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def build_dino_detr(
    model_name: str = "IDEA-Research/dino-detr-resnet-50",
    num_labels: int = 4,
    pretrained: bool = True,
    device: str = "cuda",
) -> Tuple[nn.Module, AutoImageProcessor]:
    """Build DINO-DETR model for object detection.
    
    Args:
        model_name: Hugging Face model name
        num_labels: Number of object classes (including background)
        pretrained: Whether to load pretrained weights
        device: Device to load model on
        
    Returns:
        Tuple of (model, image_processor)
    """
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Load model
    if pretrained:
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        model = AutoModelForObjectDetection.from_config(config)
    
    model = model.to(device)
    return model, image_processor


def build_teacher_student_models(
    teacher_model_name: str = "IDEA-Research/dino-detr-resnet-50",
    student_model_name: str = "IDEA-Research/dino-detr-resnet-50",
    num_labels: int = 4,
    device: str = "cuda",
) -> Tuple[nn.Module, nn.Module, AutoImageProcessor]:
    """Build teacher and student models for knowledge distillation.
    
    Args:
        teacher_model_name: Hugging Face model name for teacher
        student_model_name: Hugging Face model name for student
        num_labels: Number of object classes
        device: Device to load models on
        
    Returns:
        Tuple of (teacher_model, student_model, image_processor)
    """
    # Build teacher model (pretrained)
    teacher_model, image_processor = build_dino_detr(
        model_name=teacher_model_name,
        num_labels=num_labels,
        pretrained=True,
        device=device,
    )
    teacher_model.eval()  # Set to eval mode
    for param in teacher_model.parameters():
        param.requires_grad = False  # Freeze teacher
    
    # Build student model (from scratch or pretrained)
    student_model, _ = build_dino_detr(
        model_name=student_model_name,
        num_labels=num_labels,
        pretrained=False,  # Start from scratch for distillation
        device=device,
    )
    
    return teacher_model, student_model, image_processor


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze the backbone of the model.
    
    Args:
        model: DINO-DETR model
        freeze: Whether to freeze the backbone
    """
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        for param in model.model.backbone.parameters():
            param.requires_grad = not freeze

