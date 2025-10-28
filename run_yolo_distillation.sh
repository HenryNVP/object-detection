#!/bin/bash

# 🚀 YOLO Knowledge Distillation Launcher
# Quick and easy way to run YOLO distillation on KITTI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║         🚀 YOLO Knowledge Distillation Pipeline 🚀         ║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python found: $(python3 --version)"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠${NC}  No GPU detected - training will be slow on CPU"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Installation & Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Check if ultralytics is installed
if python3 -c "import ultralytics" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Ultralytics already installed"
else
    echo -e "${YELLOW}📦 Installing Ultralytics and dependencies...${NC}"
    pip install -q ultralytics torch torchvision tqdm pyyaml matplotlib pandas Pillow
    echo -e "${GREEN}✓${NC} Installation complete"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Default settings
MAX_SAMPLES=1000
BATCH_SIZE=16
EPOCHS=30

# Parse command line arguments
QUICK_MODE=false
FULL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            MAX_SAMPLES=500
            EPOCHS=20
            shift
            ;;
        --full)
            FULL_MODE=true
            MAX_SAMPLES=-1
            EPOCHS=50
            shift
            ;;
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick          Quick test (500 samples, 20 epochs) - 15 min"
            echo "  --full           Full dataset (all samples, 50 epochs) - 2-3 hours"
            echo "  --samples N      Use N samples (default: 1000)"
            echo "  --epochs N       Train for N epochs (default: 30)"
            echo "  --batch N        Batch size (default: 16)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                    # Quick test"
            echo "  $0 --full                     # Full training"
            echo "  $0 --samples 2000 --epochs 40 # Custom"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}⚡ Quick Mode${NC} (for testing)"
elif [ "$FULL_MODE" = true ]; then
    echo -e "${GREEN}🎯 Full Mode${NC} (for best results)"
else
    echo -e "${BLUE}⚙️  Standard Mode${NC}"
fi

echo ""
echo "📊 Training Configuration:"
echo "  • Samples:    $MAX_SAMPLES"
echo "  • Epochs:     $EPOCHS"
echo "  • Batch size: $BATCH_SIZE"
echo "  • Models:     Teacher=YOLOv8m, Student=YOLOv8n"
echo ""

# Estimate time
if [ "$QUICK_MODE" = true ]; then
    echo -e "⏱️  Estimated time: ${YELLOW}15-20 minutes${NC}"
elif [ "$FULL_MODE" = true ]; then
    echo -e "⏱️  Estimated time: ${YELLOW}2-3 hours${NC}"
else
    echo -e "⏱️  Estimated time: ${YELLOW}30-45 minutes${NC}"
fi

echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Running Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create Python script with parameters
cat > /tmp/run_yolo_distillation.py << EOF
import sys
sys.path.append('.')

# Override CONFIG
from notebooks.yolo_distillation_simple import CONFIG
CONFIG['max_samples'] = $MAX_SAMPLES
CONFIG['epochs_teacher'] = $EPOCHS
CONFIG['epochs_student'] = $EPOCHS
CONFIG['batch_size'] = $BATCH_SIZE

# Run pipeline
from notebooks.yolo_distillation_simple import main
baseline, teacher, student, results = main()

print("")
print("="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
EOF

# Run the pipeline
python3 /tmp/run_yolo_distillation.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                            ║${NC}"
    echo -e "${GREEN}║                   ✅ SUCCESS! ✅                            ║${NC}"
    echo -e "${GREEN}║                                                            ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "📁 Results saved to: ./output/yolo_distillation/"
    echo ""
    echo "📊 View results:"
    echo "  • Training curves: ./output/yolo_distillation/*/results.png"
    echo "  • Comparison plot: ./output/yolo_distillation/comparison.png"
    echo "  • Model weights:   ./output/yolo_distillation/*/weights/best.pt"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Check the comparison plot"
    echo "  2. Test inference on new images"
    echo "  3. Export to ONNX for deployment"
    echo ""
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}║                   ❌ FAILED ❌                              ║${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  • Out of memory → Use --batch 8 or --batch 4"
    echo "  • Dataset not found → Check KITTI path in config"
    echo "  • CUDA error → Update GPU drivers"
    echo ""
    exit 1
fi

# Cleanup
rm -f /tmp/run_yolo_distillation.py

