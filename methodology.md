---
layout: default
title: Methodology
---

# Detailed Methodology

## 1. Image Acquisition

- Resolution: 2048×2048
- Controlled LED lighting at 45°

## 2. Preprocessing

- Gaussian Blur
- CLAHE
- Bilateral Filtering
- Intensity normalization

## 3. Segmentation

- Otsu Thresholding
- Morphological Opening & Closing
- Connected Component Analysis

## 4. Feature Extraction

### Texture Features
- Local Binary Patterns (LBP)
- Multi-scale Gabor Filters

### Shape Features
- Area
- Perimeter
- Circularity
- Aspect Ratio

### Statistical Features
- Mean
- Variance
- Skewness
- Kurtosis

## 5. Classification

- PCA (95% variance retention)
- LDA
- Nearest Centroid
- Random Forest
