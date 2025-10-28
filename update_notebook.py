#!/usr/bin/env python3
"""Script to update distillation.ipynb with early stopping implementation."""

import json
from pathlib import Path

def update_notebook():
    """Update the notebook with early stopping and test split."""
    
    notebook_path = Path("notebooks/distillation.ipynb")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Update cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # Update 1: Add test dataset loading (Cell with "Loading datasets...")
        if 'print("Loading datasets...")' in source and 'train_dataset = build_kitti_coco_dataset' in source:
            print(f"Updating cell {i}: Adding test dataset loading...")
            cell['source'] = [
                'print("Loading datasets...")\n',
                'print("📊 Data Split Strategy:")\n',
                'print("  • Train (60%): For training teacher & student")\n',
                'print("  • Val (20%): For early stopping & monitoring during training")\n',
                'print("  • Test (20%): For final evaluation ONLY (never seen during training)")\n',
                'print()\n',
                '\n',
                '# Define transforms to convert PIL images to tensors\n',
                'import torchvision.transforms as T\n',
                '\n',
                'def get_transform():\n',
                '    """Basic transform to convert PIL images to tensors."""\n',
                '    return T.Compose([\n',
                '        T.ToTensor(),\n',
                '    ])\n',
                '\n',
                '# Load all three splits\n',
                'train_dataset = build_kitti_coco_dataset(\n',
                '    split=\'train\',\n',
                '    data_root=CONFIG[\'data_root\'],\n',
                '    transforms=None,  # We\'ll use image_processor in trainer\n',
                ')\n',
                '\n',
                'val_dataset = build_kitti_coco_dataset(\n',
                '    split=\'val\',\n',
                '    data_root=CONFIG[\'data_root\'],\n',
                '    transforms=None,\n',
                ')\n',
                '\n',
                'test_dataset = build_kitti_coco_dataset(\n',
                '    split=\'test\',\n',
                '    data_root=CONFIG[\'data_root\'],\n',
                '    transforms=None,\n',
                ')\n',
                '\n',
                '# Create data loaders\n',
                'train_loader = DataLoader(\n',
                '    train_dataset,\n',
                '    batch_size=CONFIG[\'batch_size\'],\n',
                '    shuffle=True,\n',
                '    num_workers=CONFIG[\'num_workers\'],\n',
                '    collate_fn=collate_fn,\n',
                ')\n',
                '\n',
                'val_loader = DataLoader(\n',
                '    val_dataset,\n',
                '    batch_size=CONFIG[\'batch_size\'],\n',
                '    shuffle=False,\n',
                '    num_workers=CONFIG[\'num_workers\'],\n',
                '    collate_fn=collate_fn,\n',
                ')\n',
                '\n',
                'test_loader = DataLoader(\n',
                '    test_dataset,\n',
                '    batch_size=CONFIG[\'batch_size\'],\n',
                '    shuffle=False,\n',
                '    num_workers=CONFIG[\'num_workers\'],\n',
                '    collate_fn=collate_fn,\n',
                ')\n',
                '\n',
                'print(f"✓ Train dataset: {len(train_dataset)} samples ({len(train_loader)} batches)")\n',
                'print(f"✓ Val dataset: {len(val_dataset)} samples ({len(val_loader)} batches)")\n',
                'print(f"✓ Test dataset: {len(test_dataset)} samples ({len(test_loader)} batches)")\n',
                'print(f"\\n💡 Train & val used during training, test for final evaluation only!")'
            ]
        
        # Update 2: Add early stopping to teacher training
        elif '# Fine-tune teacher model' in source and 'TEACHER_EPOCHS = 5' in source:
            print(f"Updating cell {i}: Adding early stopping to teacher training...")
            cell['source'] = [
                '# Fine-tune teacher model WITH EARLY STOPPING\n',
                'TEACHER_EPOCHS = 10  # Max epochs (early stopping may stop earlier)\n',
                'teacher_checkpoint_dir = Path(CONFIG[\'output_dir\']) / \'teacher_finetuned\'\n',
                'teacher_checkpoint_dir.mkdir(parents=True, exist_ok=True)\n',
                '\n',
                'print(f"🚀 Starting Teacher Fine-tuning (max {TEACHER_EPOCHS} epochs)")\n',
                'print(f"   Early stopping: patience=2 epochs")\n',
                'print("=" * 70)\n',
                '\n',
                '# Initialize early stopping\n',
                'early_stopping = EarlyStopping(patience=2, mode=\'min\', verbose=True)\n',
                '\n',
                'best_val_loss = float(\'inf\')\n',
                'history = {\'train_loss\': [], \'val_loss\': []}\n',
                '\n',
                'for epoch in range(1, TEACHER_EPOCHS + 1):\n',
                '    print(f"\\nEpoch {epoch}/{TEACHER_EPOCHS}")\n',
                '    print("-" * 50)\n',
                '    \n',
                '    # Train\n',
                '    train_loss = train_teacher_epoch(\n',
                '        teacher_model, train_loader, teacher_optimizer, device, epoch\n',
                '    )\n',
                '    history[\'train_loss\'].append(train_loss)\n',
                '    print(f"Train Loss: {train_loss:.4f}")\n',
                '    \n',
                '    # Validate\n',
                '    val_loss = validate_teacher(teacher_model, val_loader, device)\n',
                '    history[\'val_loss\'].append(val_loss)\n',
                '    print(f"Val Loss: {val_loss:.4f}")\n',
                '    \n',
                '    # Save best model\n',
                '    if val_loss < best_val_loss:\n',
                '        best_val_loss = val_loss\n',
                '        best_checkpoint_path = teacher_checkpoint_dir / \'best.pth\'\n',
                '        torch.save({\n',
                '            \'epoch\': epoch,\n',
                '            \'model_state_dict\': teacher_model.state_dict(),\n',
                '            \'optimizer_state_dict\': teacher_optimizer.state_dict(),\n',
                '            \'val_loss\': val_loss,\n',
                '        }, best_checkpoint_path)\n',
                '        print(f"✓ Saved best checkpoint: {best_checkpoint_path}")\n',
                '    \n',
                '    # Save epoch checkpoint\n',
                '    epoch_checkpoint_path = teacher_checkpoint_dir / f\'epoch_{epoch}.pth\'\n',
                '    torch.save({\n',
                '        \'epoch\': epoch,\n',
                '        \'model_state_dict\': teacher_model.state_dict(),\n',
                '        \'optimizer_state_dict\': teacher_optimizer.state_dict(),\n',
                '        \'val_loss\': val_loss,\n',
                '    }, epoch_checkpoint_path)\n',
                '    \n',
                '    # Check early stopping\n',
                '    if early_stopping(val_loss, epoch):\n',
                '        print(f"\\n⏹️  Training stopped early at epoch {epoch}/{TEACHER_EPOCHS}")\n',
                '        break\n',
                '    \n',
                '    # Step scheduler\n',
                '    teacher_scheduler.step()\n',
                '\n',
                'print("\\n" + "=" * 70)\n',
                'print(f"✅ Teacher fine-tuning complete!")\n',
                'print(f"Training stopped at epoch: {epoch}/{TEACHER_EPOCHS}")\n',
                'print(f"Best val loss: {best_val_loss:.4f} (epoch {early_stopping.best_epoch})")\n',
                'print(f"Checkpoints saved to: {teacher_checkpoint_dir}")\n',
                '\n',
                '# Load best model\n',
                'print("\\nLoading best checkpoint...")\n',
                'checkpoint = torch.load(teacher_checkpoint_dir / \'best.pth\')\n',
                'teacher_model.load_state_dict(checkpoint[\'model_state_dict\'])\n',
                'teacher_model.eval()\n',
                'for param in teacher_model.parameters():\n',
                '    param.requires_grad = False\n',
                'print("✓ Best model loaded and frozen for distillation")'
            ]
        
        # Update 3: Evaluate teacher on test set
        elif 'teacher_predictions = evaluate_model(teacher_model, val_loader' in source:
            print(f"Updating cell {i}: Teacher evaluation to use test set...")
            source_lines = cell['source']
            new_source = []
            for line in source_lines:
                if 'teacher_predictions = evaluate_model(teacher_model, val_loader' in line:
                    new_source.append('print("🔍 Evaluating teacher on TEST set (unseen data)...")\n')
                    new_source.append('teacher_predictions = evaluate_model(teacher_model, test_loader, device, threshold=0.5)\n')
                elif 'instances_val.json' in line and 'ann_file' in line:
                    new_source.append('    ann_file = Path(CONFIG[\'data_root\']) / \'annotations\' / \'instances_test.json\'\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
        
        # Update 4: Add early stopping to student training
        elif 'trainer.train(num_epochs=CONFIG' in source:
            print(f"Updating cell {i}: Adding early stopping to student training...")
            cell['source'] = [
                'print(f"🚀 Starting Student Training with Knowledge Distillation")\n',
                'print(f"   Max epochs: {CONFIG[\'epochs\']}")\n',
                'print(f"   Early stopping: patience=2 epochs")\n',
                'print("=" * 70)\n',
                '\n',
                '# Initialize early stopping for student\n',
                'early_stopping_student = EarlyStopping(patience=2, mode=\'min\', verbose=True)\n',
                '\n',
                'best_val_loss = float(\'inf\')\n',
                'MAX_EPOCHS = CONFIG[\'epochs\']\n',
                '\n',
                'for epoch in range(1, MAX_EPOCHS + 1):\n',
                '    print(f"\\nEpoch {epoch}/{MAX_EPOCHS}")\n',
                '    print("-" * 50)\n',
                '    \n',
                '    # Train one epoch\n',
                '    train_metrics = trainer.train_epoch(epoch)\n',
                '    print(f"Train Loss: {train_metrics[\'train_loss\']:.4f}")\n',
                '    print(f"  - Student Loss: {train_metrics[\'train_student_loss\']:.4f}")\n',
                '    print(f"  - Distillation Loss: {train_metrics[\'train_distill_loss\']:.4f}")\n',
                '    \n',
                '    # Validate\n',
                '    val_metrics = trainer.validate()\n',
                '    val_loss = val_metrics[\'val_loss\']\n',
                '    print(f"Val Loss: {val_loss:.4f}")\n',
                '    \n',
                '    # Save best model\n',
                '    if val_loss < best_val_loss:\n',
                '        best_val_loss = val_loss\n',
                '        trainer.save_checkpoint(epoch, val_loss, "best")\n',
                '    \n',
                '    # Save epoch checkpoint\n',
                '    trainer.save_checkpoint(epoch, val_loss, f"epoch_{epoch}")\n',
                '    \n',
                '    # Check early stopping\n',
                '    if early_stopping_student(val_loss, epoch):\n',
                '        print(f"\\n⏹️  Training stopped early at epoch {epoch}/{MAX_EPOCHS}")\n',
                '        break\n',
                '\n',
                'print("\\n" + "=" * 70)\n',
                'print(f"✅ Student training complete!")\n',
                'print(f"Training stopped at epoch: {epoch}/{MAX_EPOCHS}")\n',
                'print(f"Best val loss: {best_val_loss:.4f} (epoch {early_stopping_student.best_epoch})")\n',
                'print(f"Checkpoints saved to: {CONFIG[\'output_dir\']}")'
            ]
        
        # Update 5: Evaluate student on test set
        elif 'student_predictions = evaluate_model(student_model, val_loader' in source:
            print(f"Updating cell {i}: Student evaluation to use test set...")
            source_lines = cell['source']
            new_source = []
            for line in source_lines:
                if 'student_predictions = evaluate_model(student_model, val_loader' in line:
                    new_source.append('print("🔍 Evaluating STUDENT on TEST set (unseen data)...")\n')
                    new_source.append('student_predictions = evaluate_model(student_model, test_loader, device, threshold=threshold)\n')
                elif 'instances_val.json' in line and 'ann_file' in line:
                    new_source.append('    ann_file = Path(CONFIG[\'data_root\']) / \'annotations\' / \'instances_test.json\'\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✅ Notebook updated successfully: {notebook_path}")
    print("\nChanges made:")
    print("  1. ✓ Added test dataset and test_loader")
    print("  2. ✓ Added early stopping to teacher training (patience=2)")
    print("  3. ✓ Teacher evaluation uses test set instead of val set")
    print("  4. ✓ Added early stopping to student training (patience=2)")
    print("  5. ✓ Student evaluation uses test set instead of val set")

if __name__ == "__main__":
    update_notebook()

