# Plant Disease Detection via Deep Learning
### A Comparative Study of CNN, ResNet-50, EfficientNet-B3, and Vision Transformer Architectures with GAN-Based Augmentation

**Course Project — CS4152: Deep Learning & Neural Networks, Spring 2026**
Department of Computer Science, University of Management and Technology (UMT), Lahore

> Group project (5 members). My contributions: data preprocessing pipeline design, dataset unification/EDA support, and model evaluation methodology (metrics, confusion matrix analysis).

---

## Overview

Manual plant disease diagnosis is slow, subjective, and hard to scale — especially for smallholder farmers with limited access to agronomists. This project builds and rigorously compares four deep learning architectures for automated plant disease classification across a **unified 39-class taxonomy**, merging the PlantVillage and PlantDoc datasets to combine clean lab imagery with realistic field conditions.

## Key Results

| Model | Params | Test Accuracy | F1 (Weighted) | Rank |
|---|---|---|---|---|
| Custom CNN (from scratch) | 12.4M | 36.1% | 0.333 | #4 |
| EfficientNet-B3 | 12.2M | 92.0% | 0.920 | #3 |
| ResNet-50 | 25.6M | 96.6% | 0.966 | #2 |
| **ViT-B/16** | 86.6M | **98.6%** | **0.986** | **#1** |
| Soft-Voting Ensemble (all 4) | 136.8M | 98.6% | 0.986 | #1 |

**Headline finding:** transfer learning — not architecture novelty — is the dominant factor in performance. The from-scratch Custom CNN and ImageNet-pretrained EfficientNet-B3 have nearly identical parameter counts (12.4M vs 12.2M), yet EfficientNet-B3 outperforms the Custom CNN by **55.9 percentage points**, isolating pretraining as the key variable rather than model size.

## What's in this project

- **Unified 39-class dataset**: PlantVillage (clean, lab-controlled) + PlantDoc (real-world, field-captured), merged under one label namespace with per-class capping (1,200 images/class) to control imbalance.
- **Four-way architecture comparison**: Custom CNN, ResNet-50, EfficientNet-B3, ViT-B/16 — trained under an identical protocol (Focal Loss, AdamW, cosine-annealed LR, mixed precision, staged fine-tuning).
- **Class imbalance mitigation**: WeightedRandomSampler + Focal Loss (γ=2.0) to handle a 36× class imbalance ratio.
- **GAN-based augmentation**: A DCGAN trained to synthesize 64×64 leaf images for the two most underrepresented classes, demonstrating a viable offline augmentation pathway.
- **Soft-voting ensemble** across all four classifiers, with analysis of why ensembling did (and didn't) improve over the best single model.
- **Confusion matrix & error analysis** identifying which disease classes are most commonly confused and why.

## Tech Stack

`PyTorch` · `torchvision` · `timm` (ViT-B/16) · `Google Colab (NVIDIA T4 GPU)` · `torch.cuda.amp` (mixed precision)

## Repository Contents

- [`Plant_Disease_Detection_CS4152_Lite.ipynb`](./Plant_Disease_Detection_CS4152_Lite.ipynb) — full training/evaluation notebook
- [`plant_disease_report.pdf`](./plant_disease_report.pdf) — full write-up (methodology, results, discussion, references)
- [`Plant_Disease_Literature_Review.docx`](./Plant_Disease_Literature_Review.docx) — literature review (15 key references)

## Limitations

- Trained under a **LITE Mode** configuration (30% of each data split) due to compute constraints — absolute accuracy may shift at full scale, though relative model ranking is expected to hold.
- Some precision/recall values were read from generated plots rather than exact logs and are reported as approximate.
- The DCGAN module was evaluated on convergence/qualitative grounds only; it wasn't yet injected into classifier training.

## Future Work

- Full-scale training on the complete ~31,500-image dataset
- Direct injection of DCGAN-generated samples into minority-class training batches
- Higher-resolution (224×224) GAN synthesis
- Grad-CAM based interpretability validation across all 39 classes
- Packaging the best model for mobile/edge field deployment

---

**Authors:** Firooza Liaqat, Arooj Fatima, Samra Khalil, Faizan Altaf, Fakhar Abbas
