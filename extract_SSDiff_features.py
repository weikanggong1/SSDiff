import torch
import numpy as np
import nibabel as nib
import pandas as pd
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os
import copy
import argparse
from torch.utils.data import Dataset, DataLoader
import glob
from model_SSDiff import dMRItransformer, get_1d_sincos_pos_embed_from_grid
from dataloader import load_Nifti_data_nonlinear_withy as load_Nifti_data

def sph2cart(th, ph):
    return torch.cat([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], 1)

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
    parser.add_argument('-lr', default=1e-4, help='Learning rate')
    parser.add_argument('-batch_size', default=1, help='Batch size')
    parser.add_argument('-accumulation_steps', default=1, help='Batch size')
    parser.add_argument('-nepoch', default=2000, help='Number of epochs')
    parser.add_argument('-depth', default=12, help='Network depth')
    parser.add_argument('-embed_dim', default=1536, help='embed_dim')
    parser.add_argument('-patch_size', default=1, help='patch_size')
    parser.add_argument('-num_heads', default=16, help='Number of head')
    parser.add_argument('-resumeEP', default=0, help='Resume epoch')
    parser.add_argument('-resumeITER', default=0, help='Resume iteration')
    parser.add_argument('-input_drop', default=0.25, help='Input dropout')
    return parser

parser = cli_parser()
args = parser.parse_args('')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Let's use", torch.cuda.device_count(), "GPUs!")

batch_size = int(args.batch_size)
accumulation_steps = int(args.accumulation_steps)
num_workers = 12
lr = float(args.lr)
nepoch = int(args.nepoch)
patch_size = int(args.patch_size)
depth = int(args.depth)
embed_dim = int(args.embed_dim)
num_heads = int(args.num_heads)
resumeEP = int(args.resumeEP)
resumeITER = int(args.resumeITER)
input_drop = float(args.input_drop)


###load pheno variables
cov_all = pd.read_table('cov_data_v2.txt')
age = np.expand_dims(cov_all['21003-2.0'].to_numpy(),1)
cov_id = cov_all['eid']
print(len(cov_id))


xx = sorted(glob.glob('/datasets_dMRI/ukb_diffusion/*'))
all_id = []
for i in range(0, len(xx)):
    if os.path.exists(f'{xx[i]}/data_nonlinear_2mm.npy'):
        id1 = xx[i].replace('/datasets_dMRI/ukb_diffusion/','')
        all_id.append(id1)
print(len(all_id))
all_id = np.array(all_id)

inter, ia, ib = np.intersect1d(cov_id, all_id, return_indices=True)
all_id = all_id[ib]
age_matched = age[ia,:]


##data loader

all_id1 = all_id#[5000:]
train_id = all_id1[:30000]
test_id = all_id1[:60000]

age_matched1 = age_matched#[5000:,:]
y_train = age_matched1[:30000]
y_test = age_matched1[:60000]
mm = torch.mean(torch.FloatTensor(y_train)).to(device)
ss = torch.std(torch.FloatTensor(y_train)).to(device)
print(mm)
print(ss)
expected_error = np.nanmean(np.abs(y_test - np.nanmean(y_train)))
print(f'Expected MAE = {expected_error}')
# y_train = (y_train - mm) / ss
# y_test = (y_test - mm) / ss
data_dir = '/datasets_dMRI/ukb_diffusion/'
test_dataset = load_Nifti_data(data_dir, test_id, y_test)
ukb_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print(len(ukb_loader_test))

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


model = dMRItransformer(input_size=[50], patch_size=[patch_size], in_chans=nvoxel, out_chans=[1], embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm, mlp_time_embed=False,
                 use_checkpoint=False, conv=True, skip=False, attn_drop=0.0, proj_drop=0.0, pred_drop=0.0, brain_mask=brain_mask)

state_dic = torch.load('model.pth', 'cpu')
keys_to_remove = ['mlp_head.1.weight', 'mlp_head.1.bias', 'mlp_head.0.weight', 'mlp_head.0.bias']
state_dic = {k: v for k, v in state_dic.items() 
                      if k not in keys_to_remove}
                                           
xx, yy = model.load_state_dict(state_dic, strict=False)
print(xx)
print(yy)
model.eval()
model.to(device)

with torch.no_grad(): 
    for batch_idx, _batch in enumerate(ukb_loader_test):
        print(batch_idx)

        data_in, bvals, bvecs, y, index = _batch[0].to(device), _batch[1].to(device), _batch[2].to(device), _batch[3].to(device), _batch[4].to(device)
        
        data_in = data_in[:,:,0:1]
        bvals = bvals[0:1, :, :]
        bvecs = bvecs[0:1, :, :]

        fea = model.get_features1([data_in], bvals/1000.0, bvecs, pos_init)
        for j in range(0,len(fea)):
            feas1[index.cpu().numpy(),:] = fea[-1].cpu().numpy()

        fea = model1.get_features1([data_in], bvals/1000.0, bvecs, pos_init)
        for j in range(0,len(fea)):
            feas2[index.cpu().numpy(),:] = fea[-1].cpu().numpy()

        fea = model2.get_features1([data_in], bvals/1000.0, bvecs, pos_init)
        for j in range(0,len(fea)):
            feas3[index.cpu().numpy(),:] = fea[-1].cpu().numpy()
                
np.save('/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/gwk_44019/ukb_dMRItransformer/extracted_features_ukb_recononly_1dir_ep40.npy', feas1)
np.save('/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/gwk_44019/ukb_dMRItransformer/extracted_features_ukb_recononly_1dir_ep30.npy', feas2)
np.save('/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/gwk_44019/ukb_dMRItransformer/extracted_features_ukb_recononly_1dir_ep20.npy', feas3)




