# SIRE-segmentation

This repository contains the official implementation of ["Global Control for Local SO(3)-Equivariant Scale-Invariant Vessel Segmentation"](https://arxiv.org/abs/2403.15314) paper accepted at the Statistical Atlases and Computational Modeling of the Heart ([STACOM 2024](https://stacom.github.io/stacom2024/)) workshop held with MICCAI 2024.

## 1. Install

```
# Create virtual environment
conda create --name sire python=3.11
conda activate sire

# Install dependencies (developer mode)
python -m pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cu118 --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## 2. Overview
We propose a combination of a global controller leveraging voxel mask segmentations to provide boundary conditions for vessels of interest to a local, iterative vessel segmentation model. 
We introduce the preservation of scale- and rotational symmetries in the local segmentation model, leading to generalisation to vessels of unseen sizes and orientations. 
Combined with the global controller, this enables flexible 3D vascular model building, without additional retraining. 

![Alt text](images/pipeline-sire.png)

### 2.1. Global controller
We use coarse segmentations of `TotalSegmentator` to automatically define for each vessel pf interest a starting seed point for tracking and boundary conditions where the tracking should stop.
See `src/sire/aaa_wrapper.py` for the implementation of AAA global controller for aorta, iliac and renal arteries tracking.

### 2.2. Local vessel segmentation
Tracking and segmentation are perfomed jointly and start from the specified seed point (manual or automatic) and terminate based on the provided `StoppingCriteria` (use `RoiStoppingCriterion` for global control with `TotalSegmentator` by providing boundary conditions in form of a binary mask).

Each vessel controller is specified by `VesselConfig` which defines tracking and segmentation hyperparameters and global control (see line 127 in `src/sire/aaa_wrapper.py`).

Checkpoints for segmentation model and tracking model should be places under `src/sire/modesl/checkpoints` directory.

### 2.3. Surface reconstruction
2D contours can be reconstructed into 3D meshes by fitting an Implicit Neural Representation (INR) - see script `reconstruct_aaa.py` how to do that.
Evaluation of the validity of the reconstruction is done via euler characteristic (2 is correct).

### 2.4 Utilities
For some additional utilities see notebook `notebooks/utilities.ipynb`.