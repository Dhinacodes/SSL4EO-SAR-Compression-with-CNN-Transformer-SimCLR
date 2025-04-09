# 🛰️ SSL4EO-SAR Compression Pipeline

This repository implements a **hybrid, self-supervised learning pipeline** to compress high-dimensional **Sentinel-1 SAR (Synthetic Aperture Radar)** datacubes into compact and meaningful **1024-dimensional embeddings**. The pipeline combines classical image processing techniques, deep neural networks (CNN + Transformer), and **SimCLR contrastive learning**.

---

## 🧠 Motivation

Sentinel-1 SAR imagery is rich in spatial-temporal information but suffers from:
- Massive storage requirements (terabytes of global data)
- Redundant information across timestamps
- Complex, non-visual data (VH, VV polarizations)

🔍 **Our Goal**: Compress these datacubes into **dense, low-dimensional representations** that encode semantic structure — ideal for lightweight, scalable geospatial analytics.

---

## 🧰 Methodology Overview

### 1. 🔧 Preprocessing (Classical)

Each SAR datacube undergoes:
- **Bilateral Filtering**: Edge-preserving denoising
- **CLAHE**: Histogram equalization for local contrast enhancement
- **Sobel Edge Detection**: Highlights structural edges
- **Temporal Aggregation**: Mean over 4 timestamps
- **Spatial Downsampling**: (e.g., 256×256 → 64×64)

### 2. 🧠 Encoding (Deep Learning)

The compressed patch is encoded using:
- **CNN** (4 conv blocks): Extracts spatial features
- **Linear Projection**: Flattens to 128-D
- **Transformer Encoder** (2 layers, 8 heads): Captures global spatial/temporal context
- **Global Average Pooling**
- **Fully Connected Layer**: Final 1024-D semantic embedding

### 3. 🤝 SimCLR Contrastive Learning

The encoder is trained with **SimCLR**, a self-supervised approach:
- **Positive Pairs**: Augmented views of the same patch
- **Negative Pairs**: Other patches in the batch
- **Projection Head**: FC → ReLU → FC → 128-D
- **NT-Xent Loss**: Normalized Temperature-scaled Cross Entropy

---

## 📁 Dataset

- **SSL4EO-S12 v1.1**
- Source: [https://ssl4eo.org](https://ssl4eo.org)
- Contains Sentinel-1 Ground Range Detected (S1GRD) SAR imagery
- Each sample includes:
  - 4 timestamps
  - 2 polarizations (VV and VH)
  - Georeferenced patches (512×512 px → downsampled)


## 🏗️ Pipeline Architecture

[Raw S1GRD Datacube (VV + VH, 4 timestamps)]
        ↓
[Preprocessing: Bilateral + CLAHE + Sobel + Temporal Mean + Downsample]
        ↓
[ CNN (4 conv layers) ] → [ Linear Projection (128-D) ]
        ↓
[ Transformer Encoder (2 layers, 8 heads) ]
        ↓
[ Global Average Pooling ]
        ↓
[ Fully Connected Layer → 1024-D Embedding ]
        ↓
[ SimCLR Head (128-D, training only) ] → Contrastive Loss



