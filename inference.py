import torch
import numpy as np
import nibabel as nib
import pandas as pd
import os
import copy
import argparse
from torch.utils.data import Dataset, DataLoader
import glob
from ssdiff.model import dMRItransformer, get_1d_sincos_pos_embed_from_grid
from ssdiff.dataloader import load_Nifti_data_nonlinear_multidataset as load_Nifti_data

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
import random
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 

def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-data_dir', default='./data/', help='DWI data directory')   
    parser.add_argument('-ckpt_path', default='./ckpt/ssdiff_pretrain.pth', help='ckpt to load for inference')
    parser.add_argument('-lr', default=1e-4, help='Learning rate')
    parser.add_argument('-nepoch', default=50, help='Number of epochs')
    parser.add_argument('-depth', default=12, help='Network depth')
    parser.add_argument('-embed_dim', default=1536, help='Embed dims')
    parser.add_argument('-patch_size', default=1, help='Patch size')
    parser.add_argument('-num_heads', default=16, help='Number of heads')
    parser.add_argument('-input_drop', default=0.75, help='Input dropout ratio')
    parser.add_argument('-num_workers', default=12, help='num_workers')

    return parser

parser = cli_parser()
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Let's use", torch.cuda.device_count(), "GPUs!")

data_dir = str(args.data_dir)
ckpt_path = str(args.ckpt_path)
num_workers = int(args.num_workers)
lr = float(args.lr)
nepoch = int(args.nepoch)
patch_size = int(args.patch_size)
depth = int(args.depth)
embed_dim = int(args.embed_dim)
num_heads = int(args.num_heads)
input_drop = float(args.input_drop)


all_id=sorted(glob.glob(f'{data_dir}sub-*/data_nonlinear_2mm.npy'))
for i in range(0, len(all_id)):
    all_id[i] = all_id[i].replace('/data_nonlinear_2mm.npy','')
nsub = len(all_id)
print(f'Number of train subjects: {len(all_id)}')

dataset = load_Nifti_data(all_id)
loader = DataLoader(dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)
print(len(loader))

#brain mask
info = nib.load('MNI152_T1_2mm_brain.nii.gz')
brain_mask = info.get_fdata()
nvoxel = np.sum(brain_mask>0)
pos = np.argwhere(brain_mask>0)[:, ::-1]
base_dim = 100
x = get_1d_sincos_pos_embed_from_grid(base_dim, pos[:,0])
y = get_1d_sincos_pos_embed_from_grid(base_dim, pos[:,1])
z = get_1d_sincos_pos_embed_from_grid(base_dim, pos[:,2])
pos_init = torch.FloatTensor(np.concatenate((x,y,z), axis = 1)).to(device)

model = dMRItransformer(input_size=[50], patch_size=[patch_size], in_chans=nvoxel, out_chans=[1], embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm, mlp_time_embed=False, use_checkpoint=False, conv=True, skip=False, attn_drop=0.0, proj_drop=0.0, pred_drop=0.0, brain_mask=brain_mask)

state_dic = torch.load(ckpt_path, 'cpu')
keys_to_remove = ['mlp_head.1.weight', 'mlp_head.1.bias', 'mlp_head.0.weight', 'mlp_head.0.bias']
state_dic = {k: v for k, v in state_dic.items() 
                      if k not in keys_to_remove}
                                           
xx, yy = model.load_state_dict(state_dic, strict=False)
print(xx)
print(yy)
model.eval()
model.to(device)

feas = np.zeros((len(all_id), embed_dim))

with torch.no_grad(): 
    for batch_idx, _batch in enumerate(loader):
        print(batch_idx)
        print(all_id[batch_idx])
        data_in, bvals, bvecs, index = _batch[0].to(device), _batch[1].to(device), _batch[2].to(device), _batch[3].to(device)
        
        fea = model.get_features([data_in], bvals/1000.0, bvecs, pos_init)
        for j in range(0,len(fea)):
            feas[index.cpu().numpy(),:] = fea[-1].cpu().numpy()
        
#save results
df = pd.DataFrame(feas, columns=[f'Comp_{i+1}' for i in range(feas.shape[1])])
df.insert(0, 'eid', all_id)    
df.to_csv('./inference/all_features.csv', index=False)




