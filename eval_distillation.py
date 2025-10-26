"""Evaluate distilled DETR model on KITTI dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_kitti_coco_dataset, collate_fn
from src.models import build_detr
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate distilled DETR on KITTI"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default="./kitti_coco",
        help="Root directory of KITTI COCO dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/detr-resnet-50",
        help="Model name from Hugging Face",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=4,
        help="Number of object classes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for predictions (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu/mps)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detections",
    )
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    conf_threshold: float = 0.3,
) -> list:
    """Evaluate model and return predictions in COCO format.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        conf_threshold: Confidence threshold
        
    Returns:
        List of predictions in COCO format
    """
    model.eval()
    predictions = []
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        # Move to device
        images = [img.to(device) for img in images]
        
        # Forward pass
        outputs = model(images)
        
        # Process predictions
        for i, (output, target) in enumerate(zip(outputs, targets)):
            image_id = target["image_id"].item()
            
            # Extract predictions
            logits = output["logits"]  # [num_queries, num_classes]
            boxes = output["pred_boxes"]  # [num_queries, 4]
            
            # Get scores and labels
            scores = logits.softmax(-1)[:, :-1]  # Exclude no-object class
            labels = scores.argmax(-1)
            scores = scores.max(-1).values
            
            # Filter by confidence
            keep = scores > conf_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Convert boxes from [cx, cy, w, h] normalized to [x, y, w, h] absolute
            img_h, img_w = images[i].shape[-2:]
            boxes = boxes.cpu()
            boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w  # x1
            boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h  # y1
            boxes[:, 2] = boxes[:, 2] * img_w  # width
            boxes[:, 3] = boxes[:, 3] * img_h  # height
            
            # Add predictions
            for box, score, label in zip(boxes, scores, labels):
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label.item()) + 1,  # COCO categories start from 1
                    "bbox": box.tolist(),
                    "score": float(score.item()),
                })
    
    return predictions


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = build_kitti_coco_dataset(
        split=args.split,
        data_root=args.data_root,
        transforms=None,
    )
    print(f"Loaded {len(dataset)} samples")
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Load model
    print("Loading model...")
    model, _ = build_detr(
        model_name=args.model_name,
        num_labels=args.num_labels,
        pretrained=False,
        device=device,
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from: {args.checkpoint}")
    
    # Evaluate
    print("\nEvaluating...")
    predictions = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        conf_threshold=args.conf_threshold,
    )
    
    print(f"Generated {len(predictions)} predictions")
    
    # Save predictions
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(predictions, f)
        print(f"Saved predictions to: {args.output}")
    
    # Run COCO evaluation
    print("\nRunning COCO evaluation...")
    ann_file = args.data_root / "annotations" / f"instances_{args.split}.json"
    
    coco_gt = COCO(str(ann_file))
    
    if len(predictions) > 0:
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    else:
        print("No predictions generated!")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

