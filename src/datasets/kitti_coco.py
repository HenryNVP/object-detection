"""KITTI dataset in COCO format for object detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class KITTICocoDataset(Dataset):
    """KITTI dataset in COCO format for object detection.
    
    Args:
        root_dir: Root directory containing images
        annotation_file: Path to COCO format annotation JSON file
        transforms: Optional transforms to apply to images
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        annotation_file: str | Path,
        transforms: Optional[Any] = None,
    ):
        self.root_dir = Path(root_dir)
        self.coco = COCO(str(annotation_file))
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # KITTI class mapping (simplified)
        self.classes = ["background", "car", "person", "bicycle"]
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Load image info
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root_dir / img_info["file_name"]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prepare targets
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                # COCO format: [x, y, width, height]
                x, y, w, h = ann["bbox"]
                # Convert to [x1, y1, x2, y2]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
                areas.append(ann.get("area", w * h))
                iscrowd.append(ann.get("iscrowd", 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


def build_kitti_coco_dataset(
    split: str = "train",
    data_root: str | Path = "./kitti_coco",
    transforms: Optional[Any] = None,
) -> KITTICocoDataset:
    """Build KITTI dataset in COCO format.
    
    Args:
        split: Dataset split ('train' or 'val')
        data_root: Root directory containing the KITTI COCO dataset
        transforms: Optional transforms to apply
        
    Returns:
        KITTICocoDataset instance
    """
    data_root = Path(data_root)
    img_dir = data_root / "images" / split
    ann_file = data_root / "annotations" / f"instances_{split}.json"
    
    if not img_dir.exists():
        raise ValueError(f"Image directory not found: {img_dir}")
    if not ann_file.exists():
        raise ValueError(f"Annotation file not found: {ann_file}")
    
    return KITTICocoDataset(
        root_dir=img_dir,
        annotation_file=ann_file,
        transforms=transforms,
    )


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """Custom collate function for object detection.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Tuple of (images, targets) lists
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

