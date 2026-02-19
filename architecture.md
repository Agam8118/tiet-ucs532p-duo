---
layout: default
title: Architecture
---

# System Architecture

## Overview

The proposed framework follows a modular classical computer vision pipeline.

## Pipeline Stages

1. Image Acquisition  
2. Preprocessing  
3. Segmentation  
4. Feature Extraction  
5. Model Selection & Classification  

---

## Design Philosophy

The system separates:

- Domain-independent modules:
  - Preprocessing
  - Feature Extraction

- Domain-specific module:
  - ML Model Selection

This enables adaptation to metal and textile domains without redesigning the full pipeline.
