# Notebook Early Stopping Implementation - Cell-by-Cell Updates

## 📋 Overview
This guide shows exactly what to change in each cell of `notebooks/distillation.ipynb` to add:
1. ✅ Test dataset loading
2. ✅ Early stopping (patience=2) for teacher
3. ✅ Early stopping (patience=2) for student
4. ✅ Evaluation on test set (not val)

---

## Update 1: Cell 12 - Add Test Dataset Loading

**Find the cell that contains:** `print("Loading datasets...")`

**REPLACE the entire cell with:**

```python
print("Loading datasets...")
print("📊 Data Split Strategy:")
print("  • Train (60%): For training teacher & student")
print("  • Val (20%): For early stopping & monitoring during training")
print("  • Test (20%): For final evaluation ONLY (never seen during training)")
print()

# Define transforms to convert PIL images to tensors
import torchvision.transforms as T

def get_transform():
    """Basic transform to convert PIL images to tensors."""
    return T.Compose([
        T.ToTensor(),
    ])

# Load all three splits
train_dataset = build_kitti_coco_dataset(
    split='train',
    data_root=CONFIG['data_root'],
    transforms=None,  # We'll use image_processor in trainer
)

val_dataset = build_kitti_coco_dataset(
    split='val',
    data_root=CONFIG['data_root'],
    transforms=None,
)

test_dataset = build_kitti_coco_dataset(
    split='test',
    data_root=CONFIG['data_root'],
    transforms=None,
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    collate_fn=collate_fn,
)

print(f"✓ Train dataset: {len(train_dataset)} samples ({len(train_loader)} batches)")
print(f"✓ Val dataset: {len(val_dataset)} samples ({len(val_loader)} batches)")
print(f"✓ Test dataset: {len(test_dataset)} samples ({len(test_loader)} batches)")
print(f"\n💡 Train & val used during training, test for final evaluation only!")
```

---

## Update 2: Teacher Fine-tuning Section

### Find Section 7 - "Step 2: Run Fine-tuning Training Loop"

**Find the cell that contains:** `# Fine-tune teacher model` and `TEACHER_EPOCHS = 5`

**REPLACE the entire cell with:**

```python
# Fine-tune teacher model WITH EARLY STOPPING
TEACHER_EPOCHS = 10  # Max epochs (early stopping may stop earlier)
teacher_checkpoint_dir = Path(CONFIG['output_dir']) / 'teacher_finetuned'
teacher_checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"🚀 Starting Teacher Fine-tuning (max {TEACHER_EPOCHS} epochs)")
print(f"   Early stopping: patience=2 epochs")
print("=" * 70)

# Initialize early stopping
early_stopping = EarlyStopping(patience=2, mode='min', verbose=True)

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(1, TEACHER_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{TEACHER_EPOCHS}")
    print("-" * 50)
    
    # Train
    train_loss = train_teacher_epoch(
        teacher_model, train_loader, teacher_optimizer, device, epoch
    )
    history['train_loss'].append(train_loss)
    print(f"Train Loss: {train_loss:.4f}")
    
    # Validate
    val_loss = validate_teacher(teacher_model, val_loader, device)
    history['val_loss'].append(val_loss)
    print(f"Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_checkpoint_path = teacher_checkpoint_dir / 'best.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': teacher_model.state_dict(),
            'optimizer_state_dict': teacher_optimizer.state_dict(),
            'val_loss': val_loss,
        }, best_checkpoint_path)
        print(f"✓ Saved best checkpoint: {best_checkpoint_path}")
    
    # Save epoch checkpoint
    epoch_checkpoint_path = teacher_checkpoint_dir / f'epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': teacher_model.state_dict(),
        'optimizer_state_dict': teacher_optimizer.state_dict(),
        'val_loss': val_loss,
    }, epoch_checkpoint_path)
    
    # Check early stopping
    if early_stopping(val_loss, epoch):
        print(f"\n⏹️  Training stopped early at epoch {epoch}/{TEACHER_EPOCHS}")
        break
    
    # Step scheduler
    teacher_scheduler.step()

print("\n" + "=" * 70)
print(f"✅ Teacher fine-tuning complete!")
print(f"Training stopped at epoch: {epoch}/{TEACHER_EPOCHS}")
print(f"Best val loss: {best_val_loss:.4f} (epoch {early_stopping.best_epoch})")
print(f"Checkpoints saved to: {teacher_checkpoint_dir}")

# Load best model
print("\nLoading best checkpoint...")
checkpoint = torch.load(teacher_checkpoint_dir / 'best.pth')
teacher_model.load_state_dict(checkpoint['model_state_dict'])
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
print("✓ Best model loaded and frozen for distillation")
```

---

## Update 3: Teacher Evaluation - Use TEST Split

### Find Section 7 - "Step 3: Evaluate Fine-tuned Teacher"

**Find the cell that contains:** `teacher_predictions = evaluate_model(teacher_model, val_loader`

**CHANGE:**

```python
# OLD
teacher_predictions = evaluate_model(teacher_model, val_loader, device, threshold=0.5)
```

**TO:**

```python
# NEW - Evaluate on TEST set (unseen data)
print("🔍 Evaluating teacher on TEST set (unseen data)...")
teacher_predictions = evaluate_model(teacher_model, test_loader, device, threshold=0.5)
```

**Also in the same cell, CHANGE:**

```python
# OLD
ann_file = Path(CONFIG['data_root']) / 'annotations' / 'instances_val.json'
```

**TO:**

```python
# NEW
ann_file = Path(CONFIG['data_root']) / 'annotations' / 'instances_test.json'
```

---

## Update 4: Student Training - Add Early Stopping

### Find Section 8 - "Step 2: Run Distillation Training"

**Find the cell that contains:** `trainer.train(num_epochs=CONFIG['epochs']`

**REPLACE with this manual training loop:**

```python
print(f"🚀 Starting Student Training with Knowledge Distillation")
print(f"   Max epochs: {CONFIG['epochs']}")
print(f"   Early stopping: patience=2 epochs")
print("=" * 70)

# Initialize early stopping for student
early_stopping_student = EarlyStopping(patience=2, mode='min', verbose=True)

best_val_loss = float('inf')
MAX_EPOCHS = CONFIG['epochs']

for epoch in range(1, MAX_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
    print("-" * 50)
    
    # Train one epoch
    train_metrics = trainer.train_epoch(epoch)
    print(f"Train Loss: {train_metrics['train_loss']:.4f}")
    print(f"  - Student Loss: {train_metrics['train_student_loss']:.4f}")
    print(f"  - Distillation Loss: {train_metrics['train_distill_loss']:.4f}")
    
    # Validate
    val_metrics = trainer.validate()
    val_loss = val_metrics['val_loss']
    print(f"Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trainer.save_checkpoint(epoch, val_loss, "best")
    
    # Save epoch checkpoint
    trainer.save_checkpoint(epoch, val_loss, f"epoch_{epoch}")
    
    # Check early stopping
    if early_stopping_student(val_loss, epoch):
        print(f"\n⏹️  Training stopped early at epoch {epoch}/{MAX_EPOCHS}")
        break

print("\n" + "=" * 70)
print(f"✅ Student training complete!")
print(f"Training stopped at epoch: {epoch}/{MAX_EPOCHS}")
print(f"Best val loss: {best_val_loss:.4f} (epoch {early_stopping_student.best_epoch})")
print(f"Checkpoints saved to: {CONFIG['output_dir']}")
```

---

## Update 5: Student Evaluation - Use TEST Split

### Find Section 9 - "Evaluate Student Model"

**Find the cell that contains:** `student_predictions = evaluate_model(student_model, val_loader`

**CHANGE:**

```python
# OLD
student_predictions = evaluate_model(student_model, val_loader, device, threshold=threshold)
```

**TO:**

```python
# NEW - Evaluate on TEST set (unseen data)
print("🔍 Evaluating STUDENT on TEST set (unseen data)...")
student_predictions = evaluate_model(student_model, test_loader, device, threshold=threshold)
```

**Also in the same cell, CHANGE:**

```python
# OLD
ann_file = Path(CONFIG['data_root']) / 'annotations' / 'instances_val.json'
```

**TO:**

```python
# NEW
ann_file = Path(CONFIG['data_root']) / 'annotations' / 'instances_test.json'
```

---

## ✅ Verification Checklist

After making all updates, verify:

- [ ] Cell 6: `EarlyStopping` imported ✓ (already done)
- [ ] Cell 12: `test_dataset` and `test_loader` created
- [ ] Teacher training: Uses early stopping with patience=2
- [ ] Teacher evaluation: Uses `test_loader` not `val_loader`
- [ ] Student training: Uses early stopping with patience=2  
- [ ] Student evaluation: Uses `test_loader` not `val_loader`

---

## 📊 Expected Output After Changes

**Teacher Training:**
```
🚀 Starting Teacher Fine-tuning (max 10 epochs)
   Early stopping: patience=2 epochs
======================================================================

Epoch 1/10
--------------------------------------------------
Train Loss: 5.8199
Val Loss: 6.0849
✓ Validation improved (patience reset: 0/2)

Epoch 2/10
--------------------------------------------------
Train Loss: 5.1526
Val Loss: 6.1216
⚠️ No improvement for 1 epoch(s) (patience: 1/2)

Epoch 3/10
--------------------------------------------------
Train Loss: 4.9293
Val Loss: 6.4603
⚠️ No improvement for 2 epoch(s) (patience: 2/2)

🛑 Early stopping triggered!
   Best epoch: 1
   Best val loss: 6.0849

⏹️ Training stopped early at epoch 3/10
```

**Evaluation:**
```
🔍 Evaluating teacher on TEST set (unseen data)...
Evaluating: 100% 50/50
✓ Teacher generated 2500 predictions

📊 Fine-tuned Teacher Performance:
AP@0.5:0.95 = 0.250  ← True generalization performance
```

---

## 🚀 Quick Start Steps

1. **Regenerate dataset** (if not done):
   ```bash
   python scripts/prepare_kitti_coco.py \
       --train-split 0.6 --val-split 0.2 --test-split 0.2 \
       --max-samples 1000
   ```

2. **Open notebook** in Jupyter/Colab

3. **Apply updates** from this guide (Updates 1-5)

4. **Run all cells** and verify early stopping works

5. **Check results** - should see early stopping messages

Done! 🎉

