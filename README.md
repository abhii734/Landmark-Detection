# Landmark Detection Project

A comprehensive deep learning project for landmark detection using Google Landmarks Dataset v2.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Structure

```
Landmark-Detection/
├── notebooks/
│   ├── 01_Data_Exploration.ipynb      # Dataset exploration and visualization
│   ├── 02_Data_Preprocessing.ipynb   # Data loading and preprocessing
│   ├── 03_Model_Architecture.ipynb    # CNN model architecture design
│   ├── 04_Training.ipynb              # Model training and evaluation
│   └── 05_Inference_Demo.ipynb        # Inference and visualization
├── models/
│   └── landmark_model.py              # Model definitions
├── utils/
│   ├── data_loader.py                 # Data loading utilities
│   └── helpers.py                     # Helper functions
├── data/
│   └── README.md                      # Data setup instructions
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Dataset

**Google Landmarks Dataset v2** (GLDv2) - Available on:
- [Kaggle](https://www.kaggle.com/datasets/jasperbutcher/google-landmarks-v2)
- [HuggingFace](https://huggingface.co/datasets/Alanox/google-landmark-v2)
- [Original Source](https://github.com/cvdfoundation/google-landmark)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python landmark_detection.py --mode train

# 3. Run inference
python landmark_detection.py --mode inference --image path/to/image.jpg

# 4. Or open notebooks
jupyter notebook notebooks/
```

### 1. Download Dataset
```bash
# Option 1: Kaggle API
kaggle datasets download -d jasperbutcher/google-landmarks-v2

# Option 2: HuggingFace
huggingface-cli download Alanox/google-landmark-v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Open Notebooks
```bash
jupyter notebook notebooks/
```

## Model Architecture

The project implements:
- **ResNet50** backbone with pretrained ImageNet weights
- **ArcFace** loss for metric learning
- **Triplet loss** for embedding learning
- **GPU acceleration** with CUDA support

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| 01_Data_Exploration | Explore dataset structure, statistics, and sample images |
| 02_Data_Preprocessing | Data augmentation, normalization, and batching |
| 03_Model_Architecture | CNN architecture with attention mechanisms |
| 04_Training | Training loop, validation, and checkpointing |
| 05_Inference_Demo | Demo with real-time landmark detection |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 16GB+ RAM
- 100GB+ Storage

## References

- [Google Landmarks Dataset v2 - GitHub](https://github.com/cvdfoundation/google-landmark)
- [GLDv2 Paper](https://arxiv.org/abs/2004.01804)
- [Kaggle Dataset](https://www.kaggle.com/datasets/jasperbutcher/google-landmarks-v2)
- [HuggingFace Dataset](https://huggingface.co/datasets/Alanox/google-landmark-v2)