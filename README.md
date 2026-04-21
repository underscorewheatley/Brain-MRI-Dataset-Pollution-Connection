# 🧠 Brain Tumor MRI Classification — University of Waterloo Machine Learning Project

> **Machine Learning 2026** | Tolga Acan · Amber Chen · Bikramjit Garcha · Noel Gelineau · Grace Flaman · Will Sutherland
> *April 13, 2026*

---

## 📋 Overview

This project applies deep learning to classify brain tumor types from MRI scans, and explores a secondary hypothesis linking tumor-type distributions to environmental air pollution exposure. Two pretrained CNN architectures — **EfficientNetB0** and **ResNet50** — were fine-tuned via transfer learning on a dataset of 13,375 labelled grayscale MRI images, both exceeding the ≥95% accuracy target.

A secondary unsupervised analysis uses **K-Means clustering (k=3)** on learned meningioma feature embeddings to probe intra-class subgroup structure, visualized via t-SNE.

---

## 🎯 Hypotheses

1. CNN models can classify brain tumor types from MRI scans with **≥95% accuracy**, distinguishing glioma, meningioma, pituitary tumors, and no-tumor cases.
2. Geographic/temporal distributions of classified tumor types will show a **positive directional correlation** with air pollution exposure metrics, consistent with Hvidtfeldt et al. (2025) (~10% increased meningioma risk per 5,747 UFPs/cm³). *(Plausibility check only — not a causal claim.)*

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `Group_11_-_Dataset_Group_Assignment_v2_0.ipynb` | Full pipeline notebook — preprocessing, model training, evaluation, clustering |
| `ML_Project_Summary.pdf` | Two-page project summary: problem statement, methods, results, and conclusion |
| `validation_report_resnet50.png` | ResNet50 training curves, accuracy per phase, classification report, and confusion matrix |
| `validation_report_efficientnet.png` | EfficientNetB0 training curves, accuracy per phase, classification report, and confusion matrix |
| `meningioma_subgroups_resnet50.png` | t-SNE plot of K-Means meningioma subgroup clustering using ResNet50 embeddings |
| `meningioma_subgroups_efficientnet.png` | t-SNE plot of K-Means meningioma subgroup clustering using EfficientNetB0 embeddings |

---

## 🗂️ Dataset

**Source:** [Kaggle — sabersakin/brainmri](https://www.kaggle.com/sabersakin/brainmri)

| Class | Count | Share |
|-------|-------|-------|
| Glioma | 3,985 | 29.8% |
| Meningioma | 3,295 | 24.6% |
| No Tumor | 2,500 | 18.7% |
| Pituitary | 3,595 | 26.9% |
| **Total** | **13,375** | **100%** |

**Split:** 72% train / 8% validation / 20% test (stratified)

---

## ⚙️ Preprocessing Pipeline

```
Raw MRI → Padding (512×512) → Resize (242×242) → Normalize [0,1] → Augment (on-the-fly) → CNN Model
```

- Images padded to 512×512, resized to 242×242, normalized to [0,1], converted to 3-channel RGB
- Augmentation (random flips, brightness/contrast jitter) applied on-the-fly per batch
- Memory-mapped NumPy arrays kept RAM under 5 GB on Colab and Kaggle runtimes

---

## 🤖 Models

### EfficientNetB0 *(Transfer Learning)*
- Pretrained on ImageNet; fine-tuned with **3-phase gradual unfreezing**
- BatchNormalization layers kept frozen throughout to prevent domain-shift accuracy collapse
- Trained on Google Colab GPU via custom Keras Sequence generator (streaming from memmap)

### ResNet50 *(Transfer Learning)*
- **2-phase training** on Kaggle (dual Tesla T4)
  - Phase 1: Frozen base, classification head only (50 epochs, LR=5e-4)
  - Phase 2: Top 100 layers unfrozen (35 epochs, LR=5e-6)

### CNN-TumorNet *(Custom Binary CNN)*
- 4-block convolutional network for binary tumor / no-tumor classification
- Includes **LIME explainability** for clinical interpretability
- Inspired by Rasool et al. (2025)

### K-Means Clustering *(Unsupervised)*
- Applied to Dense(256) feature embeddings for meningioma test samples (n=658)
- k=3 clusters, visualized in 2D via t-SNE
- Agreement with supervised labels measured via Adjusted Rand Index (ARI)

---

## 📊 Results

### Model Performance (Test Set, n=2,671)

| Metric | EfficientNetB0 | ResNet50 |
|--------|---------------|----------|
| **Test Accuracy** | **98.17%** | **97.79%** |
| Macro F1 | 0.9838 | 0.9795 |
| Glioma F1 | 0.9717 | 0.9661 |
| Meningioma F1 | 0.9803 | 0.9802 |
| No Tumor F1 | **1.0000** | 0.9940 |
| Pituitary F1 | 0.9811 | 0.9776 |
| Best Val Acc | 98.97% | 98.88% |

Both models comfortably exceed the ≥95% target. EfficientNetB0 leads across all four classes, with perfect No Tumor classification. Glioma was the most challenging class for both models, with most misclassifications falling to Meningioma — reflecting known T1/T2 MRI imaging overlap.

### Meningioma Subgroup Clustering

| Model | Silhouette Score |
|-------|-----------------|
| ResNet50 | 0.0823 |
| EfficientNetB0 | 0.0649 |

Soft subgroup structure was found in both models' meningioma embeddings. ResNet50 produces more visually separable t-SNE clusters. Without linked patient-level pollution records, clinical interpretation of these subgroups is not yet possible — but they confirm both models encode meaningful intra-class variation.

---

## 🔬 Epidemiological Context

The pollution correlation framework is grounded in **Hvidtfeldt et al. (2025)**, a study of ~4 million Danish adults over two decades, which found approximately 10% increased meningioma risk per 5,747 UFPs/cm³ of ultrafine particle exposure. No unified patient-level dataset linking pollution histories to MRI diagnoses was publicly available; published hazard ratios were used as a secondary validation layer only.

---

## 🚧 Limitations & Next Steps

- No unified dataset links individual patient pollution histories to MRI-confirmed diagnoses — the pollution hypothesis is a population-level plausibility check only
- Constructing or accessing a linked dataset (e.g., via national cancer registry partnerships) is the most impactful next step
- Meningioma subgroup clusters require clinical metadata to be interpretable

---

## 🛠️ Environment

| Component | Details |
|-----------|---------|
| Training (EfficientNetB0) | Google Colab GPU |
| Training (ResNet50) | Kaggle — dual Tesla T4 |
| Framework | TensorFlow / Keras |
| Key Libraries | NumPy (memmap), scikit-learn, t-SNE, LIME |
