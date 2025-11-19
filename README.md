# FewSAMNet

**Official Repository — CAIMI 2025**

This is the **official repository for _FewSAMNet_**, a hybrid **SAM-CNN** framework designed for **semi-supervised few-shot medical image segmentation** with strong **multi-institutional generalization** capability.  
The work was presented at **CAIMI 2025**.

📄 **Published Abstract:**  
https://link.springer.com/article/10.1007/s10278-025-01679-0

## 📁 Code Repository Structure
src/
├── models/
│   ├── network1.py        # Proposed FewSAMNet (Hybrid SAM–CNN architecture)
│   ├── backbones/         # Backbone components and custom network layers
│   └── layers.py          # Helper layers, attention blocks, fusion modules
│
├── utils/
│   └── *                  # Metrics, loss functions, augmentations, helpers
│
├── config.py              # Main configuration file — update as required
│                          # (dataset paths, hyperparameters, SAM settings)
│
└── datagen.py             # Dataset loader and preprocessing pipeline


### Root-Level Scripts
- **train.py** — Training loop for FewSAMNet  
- **test.py** — Evaluation & inference script

## 🛠 Requirements

FewSAMNet is implemented in **Python 3.8+** and **PyTorch (1.12+ or 2.x)**.

### Core Dependencies
- Python ≥ 3.8  
- PyTorch ≥ 1.12 (CUDA support recommended)  
- torchvision ≥ 0.13  
- numpy  
- scipy  
- opencv-python  
- scikit-image  
- scikit-learn  
- matplotlib  
- tqdm  

### Optional (for SAM backbone integration)
- segment-anything  
- timm ≥ 0.9.0  
- einops

## 🚀 How to Run FewSAMNet

Before running, ensure that:
- Your dataset paths are correctly set in **`src/config.py`**
- All dependencies are installed
- You have a GPU-enabled PyTorch installation (recommended)

---

### 🔧 1. Training FewSAMNet

Run the training script:

```bash
python train.py --config src/config.py
```

## 📄 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute this software with proper attribution.

For more details, see the [`LICENSE`](LICENSE) file included in the repository.

## 📬 Contact & Support

If you have questions, encounter issues, or want to request new features, feel free to open an issue.





