"""Prepare KITTI dataset in COCO format for object detection.

This script converts KITTI dataset to COCO format with train/val splits.
Based on the CMPE_KITTI.ipynb notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm


# KITTI category mapping - Using original KITTI classes
# Map KITTI classes to unified categories (keeping KITTI names)
KITTI_CLASS_MAPPING = {
    "Car": "Car",
    "Van": "Car",  # Group with Car
    "Truck": "Truck",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",  # Group with Pedestrian
    "Cyclist": "Cyclist",
    "Tram": "Tram",
    "Misc": "Misc",
}

# Define KITTI categories (no COCO mapping)
KITTI_CATEGORIES = [
    {"id": 1, "name": "Car"},
    {"id": 2, "name": "Pedestrian"},
    {"id": 3, "name": "Cyclist"},
    {"id": 4, "name": "Truck"},
    {"id": 5, "name": "Tram"},
    {"id": 6, "name": "Misc"},
]

# Create name to ID mapping for easy lookup
CAT_NAME_TO_ID = {cat["name"]: cat["id"] for cat in KITTI_CATEGORIES}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare KITTI dataset in COCO format"
    )
    parser.add_argument(
        "--kitti-root",
        type=Path,
        default="./kitti_data/training",
        help="Root directory of KITTI dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./kitti_coco",
        help="Output directory for COCO format dataset",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.6,
        help="Train split ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test split ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    return parser.parse_args()


def read_kitti_label(label_path: Path) -> List[Dict]:
    """Read KITTI label file.
    
    Args:
        label_path: Path to KITTI label file
        
    Returns:
        List of object dictionaries
    """
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            
            obj = {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(x) for x in parts[4:8]],  # [x1, y1, x2, y2]
                "dimensions": [float(x) for x in parts[8:11]],  # [h, w, l]
                "location": [float(x) for x in parts[11:14]],  # [x, y, z]
                "rotation_y": float(parts[14]),
            }
            objects.append(obj)
    
    return objects


def convert_to_coco(
    kitti_root: Path,
    output_dir: Path,
    split: str,
    image_ids: List[str],
) -> Dict:
    """Convert KITTI to COCO format for a specific split.
    
    Args:
        kitti_root: Root directory of KITTI dataset
        output_dir: Output directory for COCO format
        split: Split name ('train' or 'val')
        image_ids: List of image IDs for this split
        
    Returns:
        COCO format dictionary
    """
    image_dir = kitti_root / "image_2"
    label_dir = kitti_root / "label_2"
    
    output_image_dir = output_dir / "images" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    
    images = []
    annotations = []
    ann_id = 1
    
    for img_id, img_name in enumerate(tqdm(image_ids, desc=f"Processing {split}")):
        # Read image
        img_path = image_dir / f"{img_name}.png"
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        img = Image.open(img_path)
        width, height = img.size
        
        # Copy image to output directory
        output_img_path = output_image_dir / f"{img_name}.png"
        shutil.copy(img_path, output_img_path)
        
        # Add image info
        images.append({
            "id": img_id + 1,
            "file_name": f"{img_name}.png",
            "width": width,
            "height": height,
        })
        
        # Read labels
        label_path = label_dir / f"{img_name}.txt"
        if not label_path.exists():
            print(f"Warning: Label not found: {label_path}")
            continue
        
        objects = read_kitti_label(label_path)
        
        # Convert annotations
        for obj in objects:
            obj_type = obj["type"]
            
            # Skip DontCare and unknown classes
            if obj_type == "DontCare" or obj_type not in KITTI_CLASS_MAPPING:
                continue
            
            # Get KITTI category ID (unified class name)
            kitti_cat_name = KITTI_CLASS_MAPPING[obj_type]
            kitti_cat_id = CAT_NAME_TO_ID[kitti_cat_name]
            
            # Get bounding box [x1, y1, x2, y2]
            x1, y1, x2, y2 = obj["bbox"]
            
            # Convert to COCO format [x, y, width, height]
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Skip invalid boxes
            if bbox_w <= 0 or bbox_h <= 0:
                continue
            
            # Add annotation
            annotations.append({
                "id": ann_id,
                "image_id": img_id + 1,
                "category_id": kitti_cat_id,
                "bbox": [x1, y1, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0,
            })
            ann_id += 1
    
    # Create COCO-style format dictionary (using KITTI class names)
    coco_dict = {
        "info": {
            "description": f"KITTI {split} set in COCO format (KITTI class names)",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": KITTI_CATEGORIES,
    }
    
    return coco_dict


def main():
    """Main function."""
    args = parse_args()
    
    print("KITTI to COCO Converter")
    print("-" * 50)
    print(f"KITTI root: {args.kitti_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Train split: {args.train_split}")
    print(f"Val split: {args.val_split}")
    print(f"Test split: {args.test_split}")
    print(f"Random seed: {args.seed}")
    
    # Validate splits sum to 1.0
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 0.01:
        raise ValueError(f"Splits must sum to 1.0, got {total_split:.2f}")
    
    print("-" * 50)
    
    # Check if KITTI dataset exists
    if not args.kitti_root.exists():
        raise ValueError(f"KITTI root not found: {args.kitti_root}")
    
    image_dir = args.kitti_root / "image_2"
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    
    # Get all image IDs
    image_files = sorted(list(image_dir.glob("*.png")))
    image_ids = [f.stem for f in image_files]
    
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]
    
    print(f"Found {len(image_ids)} images")
    
    # Split into train, val, and test
    random.seed(args.seed)
    random.shuffle(image_ids)
    
    train_idx = int(len(image_ids) * args.train_split)
    val_idx = int(len(image_ids) * (args.train_split + args.val_split))
    
    train_ids = image_ids[:train_idx]
    val_ids = image_ids[train_idx:val_idx]
    test_ids = image_ids[val_idx:]
    
    print(f"Train samples: {len(train_ids)} ({len(train_ids)/len(image_ids)*100:.1f}%)")
    print(f"Val samples: {len(val_ids)} ({len(val_ids)/len(image_ids)*100:.1f}%)")
    print(f"Test samples: {len(test_ids)} ({len(test_ids)/len(image_ids)*100:.1f}%)")
    print()
    
    # Convert train split
    print("Converting train split...")
    train_coco = convert_to_coco(
        kitti_root=args.kitti_root,
        output_dir=args.output_dir,
        split="train",
        image_ids=train_ids,
    )
    
    # Save train annotations
    ann_dir = args.output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    train_ann_path = ann_dir / "instances_train.json"
    with open(train_ann_path, "w") as f:
        json.dump(train_coco, f)
    print(f"Saved train annotations: {train_ann_path}")
    print(f"  - Images: {len(train_coco['images'])}")
    print(f"  - Annotations: {len(train_coco['annotations'])}")
    print()
    
    # Convert val split
    print("Converting val split...")
    val_coco = convert_to_coco(
        kitti_root=args.kitti_root,
        output_dir=args.output_dir,
        split="val",
        image_ids=val_ids,
    )
    
    # Save val annotations
    val_ann_path = ann_dir / "instances_val.json"
    with open(val_ann_path, "w") as f:
        json.dump(val_coco, f)
    print(f"Saved val annotations: {val_ann_path}")
    print(f"  - Images: {len(val_coco['images'])}")
    print(f"  - Annotations: {len(val_coco['annotations'])}")
    print()
    
    # Convert test split
    print("Converting test split...")
    test_coco = convert_to_coco(
        kitti_root=args.kitti_root,
        output_dir=args.output_dir,
        split="test",
        image_ids=test_ids,
    )
    
    # Save test annotations
    test_ann_path = ann_dir / "instances_test.json"
    with open(test_ann_path, "w") as f:
        json.dump(test_coco, f)
    print(f"Saved test annotations: {test_ann_path}")
    print(f"  - Images: {len(test_coco['images'])}")
    print(f"  - Annotations: {len(test_coco['annotations'])}")
    print()
    
    print("✅ Conversion complete!")
    print(f"\nOutput directory: {args.output_dir}")
    print("Directory structure:")
    print(f"{args.output_dir}/")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  └── annotations/")
    print("      ├── instances_train.json")
    print("      ├── instances_val.json")
    print("      └── instances_test.json")


if __name__ == "__main__":
    main()

