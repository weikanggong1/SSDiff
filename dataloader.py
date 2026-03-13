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
class load_Nifti_data_nonlinear_multidataset(Dataset):
    def __init__(self, ids, multishell=False):

        # self.data_dir = '/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/public/ukb_diffusion/'
        self.id = ids
        # self.even_order = [ 0,  4,  5,  6 , 7,  8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        #                 45, 46, 47, 48, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
        self.multishell = multishell

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):

        ##
        data_file = self.id[index]
        bval_file = self.id[index].replace('data_nonlinear_2mm.npy','bvals.npy')
        bvec_file = self.id[index].replace('data_nonlinear_2mm.npy','bvecs.npy')
        warp_file = self.id[index].replace('data_nonlinear_2mm.npy','warp_nonlin_diffusion_2mm.npy')
        grad_nonlin_file = self.id[index].replace('data_nonlinear_2mm.npy','grad_nonlin_2mm.npy')

        ##load diffusion data
        X = np.load(data_file) ## voxel x shells
        ## load bv, bvecs
        bvals = torch.FloatTensor(np.load(bval_file)).unsqueeze(1)
        bvecs = torch.FloatTensor(np.load(bvec_file))
        bvecs = bvecs / torch.linalg.norm(bvecs, dim=1, keepdim=True) ##shell x 3
        warp = torch.FloatTensor(np.load(warp_file)[:, 3:]) ##voxel x 9
        ## grad nonlinear
        if os.path.exists(grad_nonlin_file):
            grad_nonlin = torch.FloatTensor(np.load(grad_nonlin_file)) ##voxel x 9
            grad_nonlin = grad_nonlin.reshape(grad_nonlin.shape[0], 3, 3) + torch.eye(3).unsqueeze(0)

        ##find b0
        indx1 = bvals[:,0]<50
        ## if has multishell and user require multishell input, then pick b=1000 and b=2000 with equal prob
        has_multishell = torch.sum((bvals[:,0]>=1050) & (bvals[:,0]<=2050))
        if has_multishell>0 and self.multishell:

            # prob = np.random.rand(1)
            # if prob<0.5:
            #     indx2 = (bvals[:,0]>=1050) & (bvals[:,0]<=2050)
            # else:
            indx2 = (bvals[:,0]>=100)

        else: #else we just input the b=1000 shells
            indx2 = (bvals[:,0]>=950) & (bvals[:,0]<=1050)
            
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
        try:
            R = torch.matmul(torch.linalg.inv(sqrtmh(torch.matmul(local_warp,local_warp.mT))), local_warp) ##voxelx3x3
        except:
            R = torch.zeros(warp.shape[0], 3, 3) + torch.eye(3).unsqueeze(0)
            print(data_file)
        ##if gradnonlinear exist, correct it first
        if os.path.exists(grad_nonlin_file):
            bvecs_local = torch.matmul(bvecs, grad_nonlin)
            bvecs_local = bvecs_local / torch.linalg.norm(bvecs_local, dim=2, keepdim=True) ##voxel x shell x 3
            bvecs_local = torch.matmul(bvecs_local, R) ##voxel x shell x 3
        else:
            bvecs_local = torch.matmul(bvecs, R) ##voxel x shell x 3
        ##normalize bvecs
        bvecs_local = bvecs_local / torch.linalg.norm(bvecs_local, dim=2, keepdim=True)
        bvecs_local[torch.isnan(bvecs_local)] = 0
        bvecs_local[torch.isinf(bvecs_local)] = 0
        ##recon using b1000 and SH       
        # B, C, H = sp_harm_param.shape
        # design = rsh_cart(bvecs_local)[:, :, self.even_order] ##voxel x shell x 45
        # beta = torch.matmul()
        # out = torch.matmul(design, sp_harm_param)
        indx_perm = torch.randperm(bvals.shape[0])
        
        return X[:, indx_perm].contiguous(), bvals[indx_perm, :].unsqueeze(-1).contiguous(), bvecs_local[:, indx_perm, :].permute(1, 0, 2).contiguous(), index
