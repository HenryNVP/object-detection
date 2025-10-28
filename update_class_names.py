#!/usr/bin/env python3
"""Update class names in notebook from COCO to KITTI."""

import json
from pathlib import Path

def update_class_names():
    """Update CLASS_NAMES to use KITTI class names instead of COCO."""
    
    notebook_path = Path("notebooks/distillation.ipynb")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find and update CLASS_NAMES
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # Find CLASS_NAMES definition
        if "CLASS_NAMES = ['person', 'car', 'truck', 'bicycle', 'train', 'other']" in source:
            print(f"Found CLASS_NAMES at cell {i}, updating to KITTI class names...")
            
            # Replace the old line with new KITTI class names
            new_source = []
            for line in cell['source']:
                if "CLASS_NAMES = ['person', 'car', 'truck', 'bicycle', 'train', 'other']" in line:
                    new_source.append("CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Tram', 'Misc']  # KITTI class names\n")
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            print("✅ Updated CLASS_NAMES to KITTI class names")
            break
    else:
        print("⚠️  CLASS_NAMES not found in notebook")
        return
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"✅ Notebook saved: {notebook_path}")

if __name__ == "__main__":
    update_class_names()

