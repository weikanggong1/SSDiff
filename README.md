# SSDiff

**A foundation model for efficient data-driven characterization of brain microstructure from diffusion MRI**

## Getting Started

```bash
git clone https://github.com/weikanggong1/SSDiff
cd SSDiff-main
pip install -r requirements.txt
```

## Data preprocessing
We used FSL for data preprocessing. Ensure that you have installed FSL v6.0.7.4 or above in your Linux system. You have put your data in the /data/ folder, each folder represent a subject, with three files: data.nii.gz, bvec, bval. The example data can be downloaded [here](). The data structure is:
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
The preprocessing runs a standard dtifit to generate FA, and used FA to register your dwi data. Ensure that you have run all necessary steps before dtifits, such as eddy, topup, data denoising etc. i.e., the data.nii.gz is ready for dtifit. After it, it will warp the dwi data into stadnard MNI152 2mm space and all necessary files required to train and inference using SSDiff. When you see warp_nonlin_diffusion_2mm.npy and data_nonlinear_2mm.npy generated, the preprocessing finished.

## Model Training
Start model training (DDP supported) with the following command:
```bash
torchrun --nproc-per-node=1 train.py
```
The training loss and ckpt are saved in the ckpt folder.

## Model Evaluation
Download the pretrained ckpt [here](), and run the inference script to extract 1536 features per data.

```bash
python inference.py
```
The results are saved in the inference folder as a csv file. The first column is the subject ids and the remaining 1536 columns are the features.

Key options:

```
data_dir is the directory of the preprocessed data
ckpt_path is the path of the pretrained ckpt.
```
