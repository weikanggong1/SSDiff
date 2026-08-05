import os
import glob
import numpy as np
import torch
import argparse
import nibabel as nib
from joblib import Parallel, delayed

def round_bvals(bvals,tol=20):
    if tol == 0:
        tol = 1
    return np.round(bvals/tol)*tol  
    
##normalize bvecs
def check_bvecs(bvecs, bvals):
    if bvecs.shape[0]<bvecs.shape[1]:
        bvecs = bvecs.T
    bvecs[bvals<10,:] = [1,0,0]  
    bvecs = bvecs / np.linalg.norm(bvecs,axis=1,keepdims=True) 
    return bvecs
    
def data_to_X(data,mask):
    if len(data.shape)==3:
        return  data.flatten()[mask.flatten()>0]
    else:
        return np.reshape(data,(-1,data.shape[-1]))[mask.flatten()>0,:]
        
def downsample_warp(inp,mod = 'trilinear'):
    d2 = np.expand_dims(np.expand_dims(inp,0),0)
    out= (torch.nn.functional.interpolate(torch.FloatTensor(d2),(91,109,91), mode = mod)).numpy()[0,0,:,:,:]
    return out            
    
def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-data_dir', default='./data/')
    parser.add_argument('-ncpu', default=5)

    return parser

parser = cli_parser()
args = parser.parse_args()
data_dir = str(args.data_dir)
ncpu = int(args.ncpu)

ref_2mm = 'MNI152_T1_2mm_brain.nii.gz'
filenames=sorted(glob.glob(f'{data_dir}sub-*'))
nsub = len(filenames)
print(f'N={nsub} folders found..')


def preprocess_dwi_fsl(i, data_dir, filenames, ref_2mm):
    ff=os.path.join(filenames[i],'')
    id = filenames[i].replace(data_dir,'')    
    out_dir = data_dir
    
    dwi = glob.glob(f'{out_dir}/{id}/data.nii.gz')[0]
    bval = glob.glob(f'{out_dir}/{id}/bval')[0]
    bvec = glob.glob(f'{out_dir}/{id}/bvec')[0]
        
    target_dir = f'{out_dir}/{id}/FA/'

    ##perform dti fit
    os.system(f'rm -r {target_dir}')
    os.system(f'mkdir {target_dir}')
    os.system(f'bet {dwi} {out_dir}/{id}/data_brain -m -R -F -f 0.3')
    os.system(f'dtifit -k {out_dir}/{id}/data_brain -o {out_dir}/{id}/dti -m {out_dir}/{id}/data_brain_mask.nii.gz -r {bvec} -b {bval}')

    os.system(f'cp {out_dir}/{id}/dti_FA.nii.gz {target_dir}')    
    os.system(f'cd {target_dir} && unset SGE_ROOT && tbss_1_preproc *.nii.gz')
    os.system(f'cd {target_dir} && unset SGE_ROOT && tbss_2_reg -T')
    os.system(f'cd {target_dir} && unset SGE_ROOT && tbss_3_postreg -T')

    ##processing stage
    flirt_fname = f'{out_dir}{id}/FA/FA/dti_FA_FA_to_target.mat'
    fnirt_fname = f'{out_dir}{id}/FA/FA/dti_FA_FA_to_target_warp.nii.gz'
    bvals  = np.loadtxt(bval)
    bvals = round_bvals(bvals,tol=100.)
    bvecs  = np.loadtxt(bvec).T    
    print(bvecs.shape)
    print(bvals.shape)
    bvecs = check_bvecs(bvecs, bvals)
    
    np.save(f'{out_dir}{id}/bvals.npy', bvals)
    np.save(f'{out_dir}{id}/bvecs.npy', bvecs)
    
    if os.path.exists(f'{out_dir}/{id}/data_nonlinear_2mm.nii.gz')==0:    
        os.system(f'applywarp -i {dwi} -o {out_dir}{id}/data_nonlinear_2mm.nii.gz -w {fnirt_fname} -r {ref_2mm}')
    os.system(f'cp {fnirt_fname} {out_dir}{id}/')
    
    dMRI_data = nib.load(f'{out_dir}{id}/data_nonlinear_2mm.nii.gz').get_fdata()
    info = nib.load(ref_2mm)
    mask_2mm = info.get_fdata()    
    X = data_to_X(dMRI_data, mask_2mm)
    print(X.shape)
    np.save(f'{out_dir}{id}/data_nonlinear_2mm.npy',X)

    ##warp    
    non_warp = nib.load(fnirt_fname).get_fdata()
    print(non_warp.shape)
    info = nib.load(ref_2mm)
    mask_2mm = info.get_fdata()    
    xx, yy, zz, _ = non_warp.shape
    
    dxdx = - non_warp[0:(xx-1),:,:,0] + non_warp[1:xx,:,:,0]
    dydx = - non_warp[0:(xx-1),:,:,1] + non_warp[1:xx,:,:,1]
    dzdx = - non_warp[0:(xx-1),:,:,2] + non_warp[1:xx,:,:,2]
    
    dxdy = non_warp[:,0:(yy-1),:,0] - non_warp[:,1:yy,:,0]
    dydy = non_warp[:,0:(yy-1),:,1] - non_warp[:,1:yy,:,1]
    dzdy = non_warp[:,0:(yy-1),:,2] - non_warp[:,1:yy,:,2]
    
    
    dxdz = non_warp[:,:,0:(zz-1),0] - non_warp[:,:,1:zz,0]
    dydz = non_warp[:,:,0:(zz-1),1] - non_warp[:,:,1:zz,1]
    dzdz = non_warp[:,:,0:(zz-1),2] - non_warp[:,:,1:zz,2]
       
    things_to_downsample = [non_warp[:,:,:, 0], non_warp[:,:,:, 1], non_warp[:,:,:, 2], dxdx, dydx, dzdx, dxdy, dydy, dzdy, dxdz, dydz, dzdz]
    
    warp_out = np.zeros((np.sum(mask_2mm>0), len(things_to_downsample)))
    for ii in range(0, len(things_to_downsample)):
        out = downsample_warp(things_to_downsample[ii])
        out = data_to_X(out, mask_2mm)
        warp_out[:, ii] = out
    np.save(f'{out_dir}{id}/warp_nonlin_diffusion_2mm.npy',warp_out)

    return 0


Parallel(n_jobs=ncpu, prefer='threads')(delayed(preprocess_dwi_fsl)(i, data_dir, filenames, ref_2mm) for i in range(len(filenames)))

