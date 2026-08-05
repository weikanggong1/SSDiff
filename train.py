import torch
import numpy as np
import nibabel as nib
import pandas as pd
from ssdiff.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os
import copy
import argparse
from torch.utils.data import Dataset, DataLoader
import glob
from ssdiff.dataloader import load_Nifti_data_nonlinear_multidataset as load_Nifti_data
from ssdiff.model import dMRItransformer, get_1d_sincos_pos_embed_from_grid
import random

##ddp setting
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 


# Setup DDP
dist.init_process_group("nccl")
rank = dist.get_rank()
device = rank % torch.cuda.device_count()
seed = seed * dist.get_world_size() + rank
torch.manual_seed(seed)
torch.cuda.set_device(device)
print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-data_dir', default='./data/', help='DWI data directory')    
    parser.add_argument('-lr', default=1e-4, help='Learning rate')
    parser.add_argument('-batch_size', default=1, help='Batch size')
    parser.add_argument('-accumulation_steps', default=64, help='Accumulation steps')
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
batch_size = int(args.batch_size)
accumulation_steps = int(args.accumulation_steps)
num_workers = int(args.num_workers)
lr = float(args.lr)
nepoch = int(args.nepoch)
patch_size = int(args.patch_size)
depth = int(args.depth)
embed_dim = int(args.embed_dim)
num_heads = int(args.num_heads)
input_drop = float(args.input_drop)


train_id=sorted(glob.glob(f'{data_dir}sub-*'))
nsub = len(train_id)
print(f'Number of train subjects: {len(train_id)}')

train_dataset = load_Nifti_data(train_id)
sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True
    )
loader_train = DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)
print(len(loader_train))

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

model = DDP(model.to(device), device_ids=[rank],find_unused_parameters=True)

nparam = 0
for p in model.parameters():
    if p.requires_grad is True:
        nparam = nparam + np.prod(p.shape)
print('Number of param = ' + str(nparam / 1000000) + 'M')
model.train()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))

lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,first_cycle_steps=int(nepoch * len(loader_train)/accumulation_steps),
                                            cycle_mult=1.0,
                                            max_lr=lr,
                                            min_lr=0.00001,
                                            warmup_steps=int(5 * len(loader_train)/accumulation_steps),
                                            gamma=1.0)
os.system(f'mkdir ./ckpt/')
loss_train = []
loss_test = []
for ep in range(0, nepoch):
    sampler.set_epoch(ep)
    
    for batch_idx, _batch in enumerate(loader_train):
        
        data_in, bvals, bvecs = _batch[0].to(device), _batch[1].to(device), _batch[2].to(device)
        context = torch.nn.functional.dropout(data_in, p=input_drop)
        # print(data_in)
        model_param, model_recon= model([context], bvals/1000.0, bvecs, pos_init)

        ##DDP with gradient accumulation
        if (batch_idx+1) % accumulation_steps == 0:
            loss = torch.mean((model_recon[:,:data_in.shape[2], :] - data_in.permute(0, 2, 1))**2)/accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                     
            optimizer.zero_grad()
            lr_scheduler.step()

        else:
            with model.no_sync():
                loss = torch.mean((model_recon[:,:data_in.shape[2], :] - data_in.permute(0, 2, 1))**2)/accumulation_steps
                loss.backward()

        if rank == 0 and batch_idx%100==0:
                print(f'Train loss : EP {ep} iter {batch_idx}: {loss * accumulation_steps}')
                loss_train.append(loss.item()* accumulation_steps )
                np.save(f'./ckpt/loss_train.npy', np.array(loss_train))
                
    if rank == 0 and ep%1 == 0:
        torch.save(model.module.state_dict(), f'./ckpt/model_ep{ep}.pth')

    model.train()
    dist.barrier()

