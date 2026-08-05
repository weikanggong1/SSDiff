import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from scipy.linalg import lstsq
import glob
import os
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

from torch import linalg

def sqrtmh(A):
    L, Q = linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

class load_Nifti_data_nonlinear_multidataset(Dataset):
    def __init__(self, ids):
        
        self.id = ids

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):

        ##
        data_file = f'{self.id[index]}/data_nonlinear_2mm.npy'
        bval_file = f'{self.id[index]}/bvals.npy'
        bvec_file = f'{self.id[index]}/bvecs.npy'
        warp_file = f'{self.id[index]}/warp_nonlin_diffusion_2mm.npy'

        ##load diffusion data
        X = np.load(data_file) ## voxel x shells
        ## load bv, bvecs
        bvals = torch.FloatTensor(np.load(bval_file)).unsqueeze(1)
        bvecs = torch.FloatTensor(np.load(bvec_file))
        bvecs = bvecs / torch.linalg.norm(bvecs, dim=1, keepdim=True) ##shell x 3
        warp = torch.FloatTensor(np.load(warp_file)[:, 3:]) ##voxel x 9

        ##find b0
        indx1 = bvals[:,0]<50
        indx2 = (bvals[:,0]>=50)

        ##normalize X 
        X = torch.FloatTensor(X)
        X = np.log(np.abs(X[:, indx2]/torch.mean(X[:,indx1], dim=1, keepdim=True)))
        X[torch.isnan(X)] = 0
        X[torch.isinf(X)] = 0
        bvals = bvals[indx2,:]
        bvecs = bvecs[indx2,:]
        ##compute local bvec
        local_warp = warp.reshape(warp.shape[0], 3, 3) + torch.eye(3).unsqueeze(0)
        local_warp = torch.linalg.inv(local_warp)
        # try:
        R = torch.matmul(torch.linalg.inv(sqrtmh(torch.matmul(local_warp,local_warp.mT))), local_warp) ##voxelx3x3
        # except:
        #     R = torch.zeros(warp.shape[0], 3, 3) + torch.eye(3).unsqueeze(0)
        #     print(data_file)

        bvecs_local = torch.matmul(bvecs, R) ##voxel x shell x 3
        ##normalize bvecs
        bvecs_local = bvecs_local / torch.linalg.norm(bvecs_local, dim=2, keepdim=True)
        bvecs_local[torch.isnan(bvecs_local)] = 0
        bvecs_local[torch.isinf(bvecs_local)] = 0

        indx_perm = torch.randperm(bvals.shape[0])
        
        return X[:, indx_perm].contiguous(), bvals[indx_perm, :].unsqueeze(-1).contiguous(), bvecs_local[:, indx_perm, :].permute(1, 0, 2).contiguous(), index
