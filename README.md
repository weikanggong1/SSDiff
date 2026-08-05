# SSDiff

**A foundation model for efficient data-driven characterization of brain microstructure from diffusion MRI**

This repository contains a clean implementation of the
**SSdiff** architecture and a runnable training pipeline on example
data.

## Getting Started

```bash
git clone https://github.com/weikanggong1/SSDiff
cd SSDiff-main
pip install -r requirements.txt
```

## Data preprocessing
We used FSL for data preprocessing. Ensure that you have installed FSL v6.0.74 or above in your system. You have put your data in the /data/ folder, each folder represent a subject, with three files: data.nii.gz, bvec, bval. The example data can be downloaded [here](). The data structure is:
```
./data/
├── sub-00001/
│   ├── data.nii.gz
│   ├── bval
│   └── bvec
├── sub-00002/
│   ├── data.nii.gz
│   └── ...
└── ...
```
Then run the following command:

```bash
python data_preprocessing.py
```
The preprocessing runs a standard dtifit to generate FA, and used FA to register your dwi data. Ensure that you have run all necessary steps before dtifits, such as eddy, topup, data denoising etc. i.e., the data.nii.gz is ready for dtifit. After it, it will warp the dwi data into stadnard MNI152 2mm space and all necessary files required to train and inference using SSDiff.

## Model Training

```bash
python train.py
```

Key options:

```

```

## Model Evaluation
```bash
python inference.py
```

Key options:

```

```
