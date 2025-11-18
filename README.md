# Transfer-Learned CNNs for Adaptive Image Steganalysis and Active Defense

DeepStegDetect is a research-driven framework that evaluates the effectiveness of fine-tuned pre-trained CNNs (such as ResNet50) for detecting adaptive steganographic noise created by algorithms including S-UNIWARD, WOW, HILL, and HUGO. It also introduces an active defense system using a lightweight honeypot-based monitoring mechanism for detecting suspicious image uploads in real time.

This repository contains implementations for dataset preparation, stego generation, training, evaluation, Grad-CAM visualization, and honeypot-based active monitoring.

---

## **1. Features**

* Adaptive steganalysis using transfer-learned CNNs
* SRM-inspired residual preprocessing
* ResNet18/50/101 support
* Multi-payload detection (0.1, 0.2, 0.4 bpp)
* Grad-CAM Explainable AI for model interpretation
* Lightweight honeypot-based active defense
* Modular training and evaluation pipeline
* Reproducible experimental framework

---

## **2. Project Structure**

```
DeepStegDetect/
│
├── checkpoints/                 # Saved model weights
├── dataset/                     # Train / Val / Test splits
│   ├── BOSSBase/               # Original cover images
│   └── SUNIWARD/               # Generated stego images
│
├── models/
│   └── modified_resnet.py       # ResNet with SRM-preprocessing
│
├── scripts/
│   ├── prepare_dataset.py       # Dataset preprocessing and splitting
│   ├── generate_stego.py        # Stego image generation
│   ├── train_resnet_transfer.py # Train ResNet models
│   ├── train_srnet_lite.py      # Train SRNet-Lite baseline
│   ├── evaluate.py              # Model evaluation
│   ├── evaluate_research.py     # Alternative evaluation script
│   ├── gradcam_visualization.py # Grad-CAM visualization
│   ├── honeypot_monitor.py      # Active defense monitoring
│   ├── python_suniward.py       # Python S-UNIWARD implementation
│   └── srm_frontend.py          # SRM filter frontend
│
├── stego_tools/                 # External steganography executables
│   └── S-UNIWARD/
│       └── S-UNIWARD.exe        # S-UNIWARD executable (Windows)
│
├── processed/                   # Processed dataset (train/val/test splits)
├── outputs/                     # Evaluation outputs (confusion matrices, etc.)
├── utils.py                     # Utility functions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## **3. Installation**

### **3.1. Prerequisites**

* Python 3.7 or higher
* CUDA-capable GPU (optional, but recommended for training)

### **3.2. Install Dependencies**

```bash
# Navigate to project directory
cd DeepStegDetect

# Install Python packages
pip install -r requirements.txt
```

**Required packages:**
- `torch`, `torchvision` - Deep learning framework
- `numpy`, `opencv-python`, `Pillow` - Image processing
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Metrics
- `tqdm` - Progress bars
- `watchdog` - File system monitoring (for honeypot)
- `scipy` - Scientific computing
- `pandas` - Data handling

### **3.3. Download BOSSBase Dataset**

1. Download BOSSBase 1.01 dataset from: http://agents.fel.cvut.cz/boss/
2. Extract all images to: `dataset/BOSSBase/`
3. The images should be in `.pgm` format

---

## **4. Dataset Preparation**

### **Step 1: Generate Stego Images**

First, generate steganographic images from cover images using S-UNIWARD:

```bash
# Generate S-UNIWARD stego images at 0.2 bpp
python scripts/generate_stego.py --algo suniward --payload 0.2 --input dataset/BOSSBase --output dataset/SUNIWARD

# Or use default paths
python scripts/generate_stego.py --algo suniward --payload 0.2
```

**Note:** The script will:
- First try to use the S-UNIWARD executable from `stego_tools/S-UNIWARD/`
- If executable is not found, it falls back to Python implementation
- For other algorithms (WOW, HILL, HUGO), you need to provide executables in `stego_tools/`

### **Step 2: Prepare Dataset Splits**

Resize images, convert to RGB, and create train/val/test splits:

```bash
python scripts/prepare_dataset.py --cover dataset/BOSSBase --stego dataset/SUNIWARD --out processed --size 224
```

**Parameters:**
- `--cover`: Path to cover images directory
- `--stego`: Path to stego images directory
- `--out`: Output directory for processed dataset
- `--size`: Image size (default: 224, recommended: 160-256)

**Output structure:**
```
processed/
├── train/
│   ├── cover/
│   └── stego/
├── val/
│   ├── cover/
│   └── stego/
└── test/
    ├── cover/
    └── stego/
```

---

## **5. Training**

### **5.1. Train ResNet Models (Transfer Learning)**

Train ResNet-18, ResNet-50, or ResNet-101 with SRM preprocessing:

```bash
# Train ResNet-50 (recommended)
python scripts/train_resnet_transfer.py --model resnet50 --data_dir processed --epochs 15 --batch_size 8 --img_size 224

# Train ResNet-18 (faster, less memory)
python scripts/train_resnet_transfer.py --model resnet18 --data_dir processed --epochs 15 --batch_size 16

# Train ResNet-101 (slower, more memory)
python scripts/train_resnet_transfer.py --model resnet101 --data_dir processed --epochs 15 --batch_size 4
```

**Parameters:**
- `--model`: Model variant (`resnet18`, `resnet50`, `resnet101`)
- `--data_dir`: Path to processed dataset (default: `processed`)
- `--epochs`: Number of training epochs (default: 15)
- `--batch_size`: Batch size (default: 8, adjust based on GPU memory)
- `--img_size`: Image size (default: 224)
- `--lr`: Learning rate (default: 1e-4)
- `--freeze_until`: Freeze layers until this point (`none`, `layer1`, `layer2`, `layer3`)

**Output:**
- Best model: `checkpoints/resnet50_best.pth`
- Last checkpoint: `checkpoints/resnet50_last.pth`

### **5.2. Train SRNet-Lite (Baseline)**

Train the lightweight SRNet-Lite model:

```bash
python scripts/train_srnet_lite.py --data_dir processed --epochs 20 --batch_size 8 --img_size 160
```

**Output:**
- Best model: `checkpoints/srnet_lite_best.pth`

---

## **6. Evaluation**

### **6.1. Evaluate Model Performance**

Evaluate a trained model on the test set:

```bash
python scripts/evaluate.py --model checkpoints/resnet50_best.pth --data_dir processed --img_size 224
```

**Output:**
- Console: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- File: `outputs/confusion_matrix.png` (visualization)
- File: `outputs/confusion_matrix_metrics.txt` (text metrics)

### **6.2. Alternative Evaluation Script**

Use the research evaluation script:

```bash
python scripts/evaluate_research.py --data_dir processed --ckpt checkpoints/resnet50_best.pth --img_size 224
```

---

## **7. Grad-CAM Visualization**

Generate attention heatmaps to visualize what the model focuses on:

```bash
python scripts/gradcam_visualization.py --img test_image.png --model checkpoints/resnet50_best.pth --img_size 224
```

**Parameters:**
- `--img`: Path to input image
- `--model`: Path to model checkpoint
- `--img_size`: Image size (default: 224)
- `--output`: Output path (default: `outputs/gradcam_output.png`)

**Output:**
- Visualization with original image, heatmap, and overlay
- Prediction label (COVER/STEGO) with confidence

---

## **8. Honeypot-Based Active Defense**

Monitor a directory for new image uploads and automatically scan them for steganography.

### **Step 1: Create Watch Directory**

```bash
mkdir honeypot_watch
```

### **Step 2: Start Monitoring**

```bash
python scripts/honeypot_monitor.py --watch honeypot_watch/ --model checkpoints/resnet50_best.pth
```

**Parameters:**
- `--watch`: Directory to monitor
- `--model`: Path to trained model checkpoint
- `--log`: Log file path (default: `honeypot_log.json`)
- `--threshold`: Confidence threshold for alerts (default: 0.5)

**How it works:**
1. Monitors the watch directory for new image files
2. Computes SHA-256 hash of each new file
3. Runs steganalysis detection
4. Logs results to JSON file
5. Alerts if suspicious content detected

**Example log entry:**
```json
{
  "timestamp": "2025-11-15T10:30:45",
  "filename": "suspicious_image.png",
  "filepath": "honeypot_watch/suspicious_image.png",
  "hash": "a1b2c3d4e5f6...",
  "prediction": "STEGO",
  "confidence": 0.87,
  "probability": 0.87,
  "alert": true
}
```

**Stop monitoring:** Press `Ctrl+C`

---

## **9. Complete Workflow Example**

Here's a complete step-by-step workflow:

### **Step 1: Setup**
```bash
cd DeepStegDetect
pip install -r requirements.txt
```

### **Step 2: Prepare Dataset**
```bash
# Generate stego images
python scripts/generate_stego.py --algo suniward --payload 0.2

# Prepare dataset splits
python scripts/prepare_dataset.py --cover dataset/BOSSBase --stego dataset/SUNIWARD --out processed --size 224
```

### **Step 3: Train Model**
```bash
# Train ResNet-50
python scripts/train_resnet_transfer.py --model resnet50 --data_dir processed --epochs 15 --batch_size 8
```

### **Step 4: Evaluate**
```bash
# Evaluate on test set
python scripts/evaluate.py --model checkpoints/resnet50_best.pth --data_dir processed
```

### **Step 5: Visualize**
```bash
# Generate Grad-CAM for a test image
python scripts/gradcam_visualization.py --img test_image.png --model checkpoints/resnet50_best.pth
```

### **Step 6: Active Defense**
```bash
# Start honeypot monitoring
python scripts/honeypot_monitor.py --watch honeypot_watch/ --model checkpoints/resnet50_best.pth
```

---

## **10. Command-Line Reference**

### **Dataset Preparation**
```bash
# Generate stego images
python scripts/generate_stego.py --algo [suniward|wow|hill|hugo|all] --payload 0.2 [--input DIR] [--output DIR]

# Prepare dataset
python scripts/prepare_dataset.py --cover COVER_DIR --stego STEGO_DIR --out OUTPUT_DIR [--size 224]
```

### **Training**
```bash
# Train ResNet
python scripts/train_resnet_transfer.py --model [resnet18|resnet50|resnet101] [--data_dir DIR] [--epochs 15] [--batch_size 8]

# Train SRNet-Lite
python scripts/train_srnet_lite.py --data_dir DIR [--epochs 20] [--batch_size 8] [--img_size 160]
```

### **Evaluation**
```bash
# Standard evaluation
python scripts/evaluate.py --model CHECKPOINT [--data_dir DIR] [--img_size 224]

# Research evaluation
python scripts/evaluate_research.py --data_dir DIR --ckpt CHECKPOINT [--img_size 224]
```

### **Visualization**
```bash
python scripts/gradcam_visualization.py --img IMAGE --model CHECKPOINT [--img_size 224] [--output PATH]
```

### **Active Defense**
```bash
python scripts/honeypot_monitor.py --watch WATCH_DIR --model CHECKPOINT [--log LOG_FILE] [--threshold 0.5]
```

---

## **11. Troubleshooting**

### **Issue: CUDA Out of Memory**
- **Solution:** Reduce `--batch_size` (e.g., from 8 to 4 or 2)
- **Alternative:** Use smaller model (ResNet18 instead of ResNet50)

### **Issue: S-UNIWARD Executable Not Found**
- **Solution:** The script will automatically fall back to Python implementation
- **Alternative:** Place S-UNIWARD executable in `stego_tools/S-UNIWARD/`

### **Issue: Import Errors**
- **Solution:** Make sure you're running scripts from the project root directory
- **Check:** Verify all dependencies are installed: `pip install -r requirements.txt`

### **Issue: Dataset Not Found**
- **Solution:** Check that paths are correct and directories exist
- **Verify:** Use absolute paths if relative paths don't work

### **Issue: Windows Path Issues**
- **Solution:** Use forward slashes `/` or raw strings `r"C:\path"`
- **Note:** Scripts are Windows-compatible (num_workers=0)

---

## **12. Research Summary**

This project evaluates:

* Transferability of ImageNet-trained CNN filters to steganalysis
* Multi-algorithm stego detection (S-UNIWARD, WOW, HILL, HUGO)
* SRM-inspired preprocessing effectiveness
* Effects of freezing strategies on transfer learning
* Grad-CAM interpretability for model understanding
* Application in real-world active defense systems

**Experimental Results:**
- Accuracy: ~51% (demonstrating the difficulty of adaptive steganography detection)
- Transfer learning helps but adaptive steganography remains challenging at low payloads (0.1 bpp)
- SRM preprocessing provides useful residual features
- Grad-CAM reveals model attention patterns

The work provides a reproducible research baseline and an integrated practical defense module.

---

## **13. Citation**

If you use this code in your research, please cite:

```bibtex
@article{deepstegdetect2025,
  title={Evaluating Transfer-Learned CNNs for Adaptive Image Steganalysis and Active Defense},
  author={Prashant Patil},
  year={2025},
  institution={RNS Institute of Technology}
}
```

---

## **14. Contact**

For queries and support:

```
Prashant Patil  
Dept. of CSE (Cyber Security)  
RNS Institute of Technology  
Email: prashantpatil23cy@rnsit.ac.in
```

---

## **15. License**

This project is for research and educational purposes.

---

## **16. Acknowledgments**

* BOSSBase dataset: http://agents.fel.cvut.cz/boss/
* PyTorch and torchvision for deep learning framework
* S-UNIWARD algorithm implementation

---

**Last Updated:** November 2025
