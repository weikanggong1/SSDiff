# SSDiff: A Foundation Model for Efficient Data-Driven Characterization of Brain Microstructure from Diffusion MRI

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SSDiff** is a transformer-based foundation model designed to extract rich feature representations from diffusion-weighted MRI (dMRI) data. It leverages a self‑supervised pretraining paradigm and operates directly on voxel-wise diffusion signals, enabling efficient and generalizable characterization of brain microstructure.

---

## 📦 Requirements

- **Linux** (tested on Ubuntu 20.04/22.04)  
- **Python** ≥ 3.10  
- **CUDA** ≥ 11.8 (recommended for GPU acceleration)  
- **FSL** ≥ 6.0.7.4 (required for data preprocessing)  
- **PyTorch** ≥ 2.5.0 (with `torch.distributed` support)  

All Python dependencies are listed in [`requirements.txt`](requirements.txt) and can be installed via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/weikanggong1/SSDiff
cd SSDiff
pip install -r requirements.txt
```

---

## 📂 Data Preparation

### Input Data Structure

Place your subject‑wise dMRI data in the `./data/` directory. Each subject folder must contain three files:

```
./data/
├── sub-00001/
│   ├── data.nii.gz    # 4D diffusion-weighted image (ready for dtifit)
│   ├── bval           # b-values (one per volume)
│   └── bvec           # b-vectors (three columns, one per volume)
├── sub-00002/
│   └── ...
└── ...
```

> **Pre‑processing prerequisite**: The input `data.nii.gz` should already have been processed with **eddy**, **topup**, and denoising (if applicable), so that it is fully prepared for `dtifit`.

### Preprocessing Pipeline

We provide a preprocessing script (`data_preprocessing.py`) that performs the following steps using FSL:

- Brain extraction (`bet`)
- Diffusion tensor fitting (`dtifit`)
- Non‑linear registration of FA to MNI152 2mm space (`tbss` pipeline)
- Warping of raw dMRI data and computation of local Jacobian‑based spatial derivatives

Run the pipeline with:

```bash
python data_preprocessing.py
```

**Outputs** (per subject, saved in the same folder):
- `data_nonlinear_2mm.npy` – warped dMRI signals in MNI space (voxels × shells)
- `bvals.npy` / `bvecs.npy` – rounded b‑values and reoriented b‑vectors
- `warp_nonlin_diffusion_2mm.npy` – spatial warping parameters (voxel‑wise deformation gradients)

Processing may take several minutes per subject depending on data size and system resources. The script runs in parallel using `joblib` (set `-ncpu` to control concurrency).

---

## 🧠 Model Training

We support **Distributed Data Parallel (DDP)** training across multiple GPUs. To start training:

```bash
torchrun --nproc-per-node=<N_GPUS> train.py
```

Key training hyperparameters can be adjusted via command‑line arguments (see `train.py` for full list). All checkpoints and loss curves are saved in the `./ckpt/` directory.

**Example** (single GPU):
```bash
torchrun --nproc-per-node=1 train.py
```

---

## 🔍 Feature Extraction & Inference

Download the pretrained model [checkpoint]() and place it in `./ckpt/ssdiff_pretrain.pth`. Then run inference to extract 1536‑dimensional feature vectors per subject:

```bash
python inference.py
```

The output CSV file (`all_features.csv`) will be saved in the `./inference/` folder. It contains one row per subject, with the first column (`eid`) listing subject identifiers and the remaining 1536 columns representing the extracted features.

**Customizing inference**:
```bash
python inference.py --data_dir /path/to/your/data --ckpt_path /path/to/checkpoint.pth
```

---

## 📝 Important Notes

- **FSL Environment**: Ensure that FSL is properly installed and the environment variables (`FSLDIR`, `PATH`, etc.) are sourced before running `data_preprocessing.py`. For example, add the following to your `~/.bashrc`:
- **GPU Memory**: The model requires a GPU with at least 24 GB of memory for full‑size inference; memory usage can be reduced by lowering `embed_dim` or `depth` for exploratory runs.
- **Reproducibility**: All random seeds are fixed in the code. However, due to non‑deterministic CUDA operations, slight numerical variations may occur.

---
## 📧 Contact

For questions or issues, please open a GitHub issue or contact the corresponding author at weikanggong@fudan.edu.cn.

