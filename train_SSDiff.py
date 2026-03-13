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
from dataloader import load_Nifti_data_nonlinear_multidataset as load_Nifti_data
from model_SSDiff import dMRItransformer, get_1d_sincos_pos_embed_from_grid

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

# Setup DDP:
dist.init_process_group("nccl")
# assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
rank = dist.get_rank()
device = rank % torch.cuda.device_count()
seed = seed * dist.get_world_size() + rank
torch.manual_seed(seed)
torch.cuda.set_device(device)
print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-lr', default=1e-4, help='Learning rate')
    parser.add_argument('-batch_size', default=1, help='Batch size')
    parser.add_argument('-accumulation_steps', default=64, help='Batch size')
    parser.add_argument('-nepoch', default=2000, help='Number of epochs')
    parser.add_argument('-depth', default=12, help='Network depth')
    parser.add_argument('-embed_dim', default=1536, help='embed_dim')
    parser.add_argument('-patch_size', default=1, help='patch_size')
    parser.add_argument('-num_heads', default=16, help='Number of head')
    parser.add_argument('-resumeEP', default=0, help='Resume epoch')
    parser.add_argument('-resumeITER', default=0, help='Resume iteration')
    parser.add_argument('-input_drop', default=0.75, help='Input dropout')

    return parser

parser = cli_parser()
args = parser.parse_args()

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

all_id = np.array(all_id)
##data loader
np.random.seed(42)
permute_indx = np.random.permutation(len(all_id))
train_id = all_id[permute_indx][:79000]
print(f'Number of train subjects: {len(train_id)}')
test_id = all_id1
print(f'Number of test subjects: {len(test_id)}')

train_dataset = load_Nifti_data(train_id, multishell=True)
sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True
    )
ukb_loader_train = DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)
print(len(ukb_loader_train))

test_dataset = load_Nifti_data(test_id, multishell=True)
test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True
    )
ukb_loader_test = DataLoader(test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True)
print(len(ukb_loader_test))

#brain mask
info = nib.load('MNI152_T1_2mm_brain.nii.gz')
brain_mask = info.get_fdata()
nvoxel = np.sum(brain_mask>0)
pos = np.argwhere(brain_mask>0)[:, ::-1]
base_dim = 100
x = get_1d_sincos_pos_embed_from_grid(base_dim, pos[:,0]) ##400x100
y = get_1d_sincos_pos_embed_from_grid(base_dim, pos[:,1]) ##
z = get_1d_sincos_pos_embed_from_grid(base_dim, pos[:,2])
pos_init = torch.FloatTensor(np.concatenate((x,y,z), axis = 1)).to(device)

model = dMRItransformer(input_size=[50], patch_size=[patch_size], in_chans=nvoxel, out_chans=[1], embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm, mlp_time_embed=False,
                 use_checkpoint=False, conv=True, skip=False, attn_drop=0.0, proj_drop=0.0, pred_drop=0.0, brain_mask=brain_mask)

model = DDP(model.to(device), device_ids=[rank],find_unused_parameters=True)

nparam = 0
for p in model.parameters():
    if p.requires_grad is True:
        nparam = nparam + np.prod(p.shape)
print('Number of param = ' + str(nparam / 1000000) + 'M')
model.train()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))

lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,first_cycle_steps=int(nepoch * len(ukb_loader_train)/accumulation_steps),
                                            cycle_mult=1.0,
                                            max_lr=lr,
                                            min_lr=0.00001,
                                            warmup_steps=int(5 * len(ukb_loader_train)/accumulation_steps),
                                            gamma=1.0)

loss_train = []
loss_test = []
for ep in range(0, nepoch):
    sampler.set_epoch(ep)

    if ep%10 == 0:
        with torch.no_grad():
            model.eval()
            feas = torch.zeros(len(test_id), embed_dim, 2).to(device)
            for batch_idx, _batch in enumerate(ukb_loader_test):

                data_in, bvals, bvecs, index = _batch[0].to(device), _batch[1].to(device), _batch[2].to(device) , _batch[3].to(device)                 
                model_param, model_recon= model([data_in], bvals/1000.0, bvecs, pos_init)
                loss = torch.mean((model_recon[:,:data_in.shape[2], :] - data_in.permute(0, 2, 1))**2)/accumulation_steps

                fea = model.module.get_features([data_in], bvals/1000.0, bvecs, pos_init)
                for j in range(0,len(fea)):
                    feas[index.cpu().numpy(),:,j] = fea[j]

                if rank == 0 and batch_idx%10==0:
                    print(f'Test loss: EP {ep} iter {batch_idx}: {loss * accumulation_steps}')

                    loss_test.append(loss.item()* accumulation_steps )
                    np.save(f'{save_path}/{save_basename}loss_test.npy', np.array(loss_test))
            dist.all_reduce(feas, op=dist.ReduceOp.SUM)    
            np.save(f'{save_path}/{save_basename}extracted_features_ukb_recononly_ep{ep}.npy', feas.cpu().numpy())

    for batch_idx, _batch in enumerate(ukb_loader_train):
        
        data_in, bvals, bvecs = _batch[0].to(device), _batch[1].to(device), _batch[2].to(device)
        context = torch.nn.functional.dropout(data_in, p=input_drop)
        # print(data_in)
        model_param, model_recon= model([context], bvals/1000.0, bvecs, pos_init)

        ##DDP with gradient accumulation
        if (batch_idx+1) % accumulation_steps == 0:
            loss = torch.mean((model_recon[:,:data_in.shape[2], :] - data_in.permute(0, 2, 1))**2)/accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
            lr_scheduler.step()

        else:
            with model.no_sync():
                loss = torch.mean((model_recon[:,:data_in.shape[2], :] - data_in.permute(0, 2, 1))**2)/accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if rank == 0 and batch_idx%100==0:
                print(f'Train loss : EP {ep} iter {batch_idx}: {loss * accumulation_steps}')

                loss_train.append(loss.item()* accumulation_steps )
                np.save(f'{save_path}/{save_basename}loss_train.npy', np.array(loss_train))
                
    if rank == 0 and ep%10 == 0:
        torch.save(model.module.state_dict(), f'{save_path}/{save_basename}model_ep{ep}.pth')

    model.train()
    dist.barrier()

