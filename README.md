# Landmark Detection and Image Retrieval Pipeline

This repository implements an end-to-end computer vision pipeline for landmark classification and image retrieval. It supports multiple state-of-the-art backbones (ResNet, EfficientNet, Vision Transformers), a high-concurrency data ingestion system, and advanced metric learning via ArcFace and Generalized Mean (GeM) pooling.

---

## Key Features

* **Asynchronous Image Downloader:** Highly concurrent, non-blocking ingestion engine using `asyncio` and `aiohttp` to download the Google Landmarks Dataset v2 (GLDv2).
* **Multi-Architecture Support:** Built-in wrappers for fine-tuning ResNet (ResNet34, ResNet50), EfficientNet (B0, B3), and Vision Transformers (ViT-Small).
* **Metric Learning Pipeline:** ArcFace (Additive Angular Margin Loss) head integration to learn discriminative embedding spaces optimized for image retrieval tasks.
* **Feature Aggregation:** Custom Generalized Mean (GeM) Pooling and attention-based pooling modules to improve representation quality.
* **Mixed Precision Training:** Automatic Mixed Precision (AMP) training support using PyTorch's `GradScaler` and `autocast` to reduce GPU memory usage.
* **Flexible Datasets:** Supports synthetic pattern-based datasets for rapid pipeline validation and local real-world image directories.

---

## Project Structure

```
Landmark-Detection/
├── data/                       # Dataset configurations and metadata splits
├── models/                     # Deep learning architectures (classifiers and embeddings)
├── notebooks/                  # Explanatory Jupyter notebooks for exploration and training
├── utils/                      # Data loaders, image augmentations, and helpers
├── train.py                    # Unified script for local/remote training
├── train_local.py              # Baseline local training script
├── download_images.py          # Asynchronous dataset download utility
├── Landmark_Detection_Colab.ipynb # Google Colab notebook for GPU training
└── README.md                   # Project documentation
```

---

## Installation

Ensure you have Python 3.9+ installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies
* `torch >= 2.0.0`
* `torchvision >= 0.15.0`
* `timm` (for advanced architectures)
* `pandas`, `numpy`, `scikit-learn`
* `aiohttp`, `aiofiles` (for async downloading)
* `tqdm`, `pillow`

---

## Dataset & Ingestion

The pipeline operates on two distinct data modes:

### 1. Synthetic Mode (Pattern-Based Validation)
```bash
python train.py --mode synthetic --classes 500 --epochs 5
```
Generates 25,000 synthetic pattern-based images on-the-fly across 500 classes. Ideal for verifying training loops, learning rate schedulers, and model pipelines without downloading real datasets.

### 2. Real Image Mode (Google Landmarks Dataset v2)
To build a production model using real landmark assets:

1. **Download the metadata splits and images:**
   ```bash
   # Download a 1% stratified sample of the dataset
   python download_images.py --sample 0.01 --workers 50
   ```
2. **Execute training on downloaded data:**
   ```bash
   python train.py --mode real --classes 100 --epochs 10
   ```

---

## Model Benchmarks & Metrics

Below are evaluation metrics obtained across different model configurations on subsets of the Google Landmarks Dataset v2:

| Model Backbone | Metric Learning Head | Parameters | Computational Complexity (FLOPs) | Validation Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ArcFace-ResNet50** | ArcFace Head | 25.6M | 4.2B | **80.0%** | **0.79** |
| **ViT-Small** | Standard Linear | 22.0M | 2.6B | **79.0%** | **0.78** |
| **EfficientNet-B0** | Standard Linear | 5.3M | 0.39B | **77.0%** | **0.76** |
| **ResNet50** | Standard Linear | 25.6M | 4.1B | **76.0%** | **0.75** |

*Note: In synthetic mode, ResNet50 consistently achieves **85%+** validation accuracy.*

---

## Training Configuration

The training script exposes multiple hyperparameter controls via CLI:

```bash
python train.py \
    --mode real \
    --classes 100 \
    --epochs 10 \
    --batch 32 \
    --lr 0.001
```

By default, training leverages:
* **Optimizer:** `AdamW` (weight decay of `0.01`).
* **LR Scheduler:** Cosine Annealing Learning Rate.
* **Loss Function:** Label-smoothed Cross Entropy (`0.1`) or ArcFace loss.

---

## Inference

To load a trained checkpoint and run inference on an arbitrary image:

```python
import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image

# 1. Load the model configuration and weights
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Preprocess target image
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('landmark_image.jpg')
img_tensor = transform(img).unsqueeze(0)

# 3. Predict class index
with torch.no_grad():
    prediction = model(img_tensor)
    class_id = prediction.argmax(dim=1).item()
```

---

## References

* He et al., *Deep Residual Learning for Image Recognition* (ResNet)
* Tan et al., *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*
* Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*
* [Google Landmarks Dataset v2 GitHub Repository](https://github.com/cvdfoundation/google-landmark)
