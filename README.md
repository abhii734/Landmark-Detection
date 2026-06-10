# 🏛️ Landmark Detection Training

**Train a ResNet50 landmark classifier in minutes - no downloads needed!**

This project matches the YouTube tutorial for landmark detection, using synthetic training data for quick experiments OR real images for production models.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![GPU](https://img.shields.io/badge/GPU-T4+-green.svg)

---

## 🚀 Quick Start (Exactly Like the Video)

### Option 1: Colab Notebook (Easiest)
1. Open `Landmark_Detection_Colab.ipynb` in Google Colab
2. **Runtime → Change runtime type → GPU**
3. Run all cells (Shift+Enter)
4. Download trained model when done

### Option 2: Local Training
```bash
# Install dependencies
pip install torch torchvision pandas numpy scikit-learn tqdm pillow

# Run training (synthetic mode - like video)
python train.py

# Run with real images
python train.py --mode real
```

---

## 📊 Training Modes

### Synthetic Mode (Default - No Downloads Needed)
```bash
python train.py --mode synthetic
```
- **500 landmark classes**
- **50 images per class = 25,000 training images**
- Images are **generated programmatically** (no downloads!)
- Matches the YouTube tutorial exactly
- ~10-15 min on GPU (T4/Colab)

### Real Image Mode (Your Downloaded Images)
```bash
python train.py --mode real --classes 100
```
- Uses images from `images_000/`
- Limited by available data (100 classes with 2+ images each)
- **8,266 images** available, ~384 usable for training

---

## 🎯 Results

| Mode | Classes | Images | GPU Training | Validation Accuracy |
|------|---------|--------|--------------|---------------------|
| Synthetic | 500 | 25,000 | ~10 min | ~85%+ |
| Real | 100 | 384 | ~5 min | Varies |

---

## 📁 Project Structure

```
Landmark-Detection/
├── train.py                    # Main training script (unified)
├── train_local.py              # Original local GPU script
├── Landmark_Detection_Colab.ipynb  # Colab notebook (like video)
├── download_images.py         # Download real images
├── train_part_*.csv          # GLDv2 metadata (all 10 parts)
├── images_000/                # Downloaded images (8,266 files)
├── checkpoints/               # Trained models saved here
└── README.md
```

---

## 🔧 Customization

```bash
# More epochs
python train.py --epochs 10

# Smaller batch size (for limited GPU memory)
python train.py --batch 32

# Fewer classes (faster training)
python train.py --classes 100

# Combine options
python train.py --mode synthetic --epochs 20 --batch 32 --classes 200
```

---

## 📥 Using Trained Models

```python
import torch
from torchvision import models

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
import torchvision.transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('your_image.jpg')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    prediction = model(img_tensor)
    class_id = prediction.argmax(1).item()
```

---

## 🔗 Resources

- [YouTube Tutorial](https://www.youtube.com/watch?v=peeJp1k-Chs)
- [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tqdm>=4.65.0
pillow>=9.0.0
```

Install with: `pip install -r requirements.txt`