# iS-GNN: Interpolation of Crustal Stress Maps using a Graph Neural Network model  
**Version 1.0.0 – Companion Code for the AGU Submission**

This repository contains the official implementation of **iS-GNN**, a symmetry-aware Graph Neural Network for interpolating the maximum horizontal stress orientation (SH$_\text{max}$) from sparse geological observations.  
The model is developed as part of the manuscript:

> **iS-GNN: Interpolation of Crustal Stress Maps using a Graph Neural Network model**  
> *K. A. Gyamfi, M. M. C. Carafa 2025 (submitted to AGU)*

---

## 🌍 Overview

Estimating SH$_{\max}$ from sparse, irregularly distributed stress observations is a persistent challenge in tectonic and geomechanical studies.  
The **iS-GNN** model addresses this by:

- encoding SH$_{\max}$ azimuths using **axial-aware trigonometric embeddings**  
- constructing **geodesy-informed graphs** based on spatial and geological proximity  
- training inductively via **masked subgraph reconstruction**, inspired by spatiotemporal kriging  
- performing **interpolation on arbitrary uniform grids**  
- supporting **nested-grid inference** for fine-resolution products (e.g., 0.2° grids)

The method is validated using the **World Stress Map 2025 (WSM25)** dataset.

---
## 📁 Repository Structure

iS-GNN/
│
├── LICENSE # Open-source license 
├── README.md # Project documentation
│
├── data/ # (Optional) Sample or processed datasets
│
├── src/ # Source code (Python package)
│ └── isgnn/
│ ├── init.py # Package initializer
│ ├── sh_utils.py # Stress & angle utilities
│ ├── sh_post_utils.py # Postprocessing utilities (errors, MAE, etc.)
│ ├── basic_structure.py # Core model building blocks
│
├── model/ # GNN models from IGNNK paper. Note used in iS-GNN
│
├── trained_models/ # Saved checkpoints from development (.pt, .pth)
│
├── publication_figs_maps/ # High-resolution maps and composites for paper
│
├── figures_maps/ # Intermediate or auxiliary map figures
│
├── python_scripts_only/ # Standalone utility scripts (non-package)
│
<<<<<<< HEAD
└── ISGNN_Final-With-PostBlend-AsFeature-Training.ipynb # Main notebook for training & experiments
=======
└── ISGNN_Final-With-PostBlend-AsFeature-Training.ipynb # Main notebook for training & experiments
>>>>>>> Update notebook and utils
