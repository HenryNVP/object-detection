# 🚀 YOLO Distillation - Ultra Quick Start

## ⚡ 3 Ways to Run (Choose One)

### 1. Fastest Way (1 command) ⭐ RECOMMENDED

```bash
./run_yolo_distillation.sh --quick
```

**Done in 15 minutes!** ✨

---

### 2. Python Script Way

```bash
cd notebooks
python yolo_distillation_simple.py
```

**Done in 30 minutes!**

---

### 3. Custom Way (Python)

```python
from notebooks.yolo_distillation_simple import *

# Convert dataset
data_yaml = convert_kitti_to_yolo()

# Train models
baseline, baseline_metrics = train_baseline(data_yaml)
teacher, teacher_metrics = train_teacher(data_yaml)
student, student_metrics = train_student_distilled(data_yaml)

# Compare
compare_results(baseline_metrics, teacher_metrics, student_metrics)
```

---

## 📁 What You Get

After running, check:

```
output/yolo_distillation/
├── baseline_yolov8n/weights/best.pt       # Baseline model
├── teacher_yolov8m/weights/best.pt        # Teacher model
├── student_yolov8n_distilled/weights/best.pt  # Distilled student
└── comparison.png                          # Results comparison
```

---

## 🎯 Expected Results

```
Baseline YOLOv8n:      mAP@0.50 ≈ 0.45  (3.2M params)
Teacher YOLOv8m:       mAP@0.50 ≈ 0.55  (25.9M params)
Student (Distilled):   mAP@0.50 ≈ 0.50  (3.2M params)

🎉 Distillation improvement: +10% over baseline!
```

---

## 🔥 Test Your Model

```python
from ultralytics import YOLO

# Load best student model
model = YOLO('output/yolo_distillation/student_yolov8n_distilled/weights/best.pt')

# Predict on image
results = model.predict('path/to/image.jpg', conf=0.25)

# Show results
results[0].show()
```

---

## 🚢 Export for Production

```python
# Export to ONNX (universal format)
model.export(format='onnx')

# Now you have: best.onnx (portable, fast)
```

---

## 💡 Pro Tips

1. **Out of memory?** Use `--batch 8`
2. **Too slow?** Use `--quick` flag
3. **Want best results?** Use `--full` flag
4. **Need help?** Check `YOLO_DISTILLATION_README.md`

---

## 📚 Full Documentation

- **Complete guide:** `YOLO_DISTILLATION_README.md`
- **YOLO vs DETR:** `YOLO_VS_DETR.md`
- **Config file:** `configs/yolo_distillation.yaml`

---

## ⚠️ Troubleshooting

### Problem: CUDA out of memory
```bash
./run_yolo_distillation.sh --batch 8
```

### Problem: Too slow
```bash
./run_yolo_distillation.sh --quick  # Use 500 samples, 20 epochs
```

### Problem: Dataset not found
```bash
# Check your KITTI path in the script:
# CONFIG['kitti_root'] = './kitti_data/training'
```

---

## 🎉 That's It!

**Literally 1 command and you're done:**

```bash
./run_yolo_distillation.sh --quick
```

Enjoy your fast, efficient YOLO models! 🚀

