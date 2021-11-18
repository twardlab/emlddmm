# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # EMLDDMM Example: MRICloud atlas data
# 
# 
# %% [markdown]
# ## Introdution
# We run registration between a pair of T1 MRI images from mricloud.org.
# Each includes a grayscale image and a set of labels.  We first run the
# registration pipeline with a config file. We then run the transformation pipeline.
# 

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import torch
import matplotlib.pyplot as plt
import emlddmm
import json
import nrrd
import os
from datetime import datetime
import nibabel as nib
import glob
from skimage import measure
from mayavi import mlab

# for debugging only
import imp
imp.reload(emlddmm)

# %% [markdown]
# ## Registration pipeline
# Here we load the atlas and target images, and run multi scale registration.
# Note the inputs
# 
# atlas_name,label_name,target_name: vtk filenames for images
# 
# config_file: json filename for options
# 
# output_dir: string containing a directory for outputs
# 
# start by converting data to vtk
directory = 'C:\\Users\\BGAdmin\\data\\Adult27-55'
files = glob.glob(os.path.join(directory,'*.img'))
for f in files:
    if 'Labels' not in f and 'MNI' not in f:
        continue
        
    print(f)

    vol = nib.load(f)
    
    dI = vol.header.get_zooms()[:3]
    I = vol.get_fdata()[...,0]
    xI = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(I.shape,dI)]    
    dtype = vol.header.get_data_dtype()
    #if 'Labels' in f:
    #    dtype = np.dtype('<u2') # 16 bit unsigned even though above says 16 bit signed
    
    # write it back out
    emlddmm.write_vtk_data(os.path.splitext(f)[0] + '.vtk',xI,I[None].astype(dtype),'mricloud_atlas')
    
    
# %%
target_name = 'C:\\Users\\BGAdmin\\data\\MD816\\HR_NIHxCSHL_50um_14T_M1_masked.vtk'
label_name = 'C:\\Users\BGAdmin\\data\\Adult27-55\\Adt27-55_02_Adt27-55_02_FullLabels.vtk'
atlas_name = 'C:\\Users\\BGAdmin\\data\\Allen_Atlas_vtk/ara_nissl_50.vtk'
config_file = 'config_mricloud.json'

time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
output_dir = 'mricloud_test_outputs_'+time


# %%
with open(config_file) as f:
    config = json.load(f)
# I'm getting this for initial downsampling for preprocessing
downIs = config['downI']
downJs = config['downJ']


# %%
plt.rcParams["figure.figsize"] = (10,10)
# atlas
imp.reload(emlddmm)
xI,I,title,names = emlddmm.read_data(atlas_name)
I = I.astype(float)
# normalize
I /= np.mean(np.abs(I))
dI = np.array([x[1]-x[0] for x in xI])
print(dI)
fig = emlddmm.draw(I,xI)
fig[0].suptitle('Atlas image')


# %%
# initial downsampling so there isn't so much on the gpu
mindownI = np.min(np.array(downIs),0)
xI,I = emlddmm.downsample_image_domain(xI,I,mindownI)
downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
dI = [x[1]-x[0] for x in xI]
print(dI)
nI = np.array(I.shape,dtype=int)
# update our config variable
config['downI'] = downIs


# %%
# target
imp.reload(emlddmm)
xJ,J,title,names = emlddmm.read_data(target_name)
J = J.astype(float)
J /= np.mean(np.abs(J))
dJ = np.array([x[1]-x[0] for x in xJ])
print(dJ)
J = J.astype(float)#**0.25
fig = emlddmm.draw(J,xJ)
fig[0].suptitle('Target image')
W0 = np.ones_like(J[0])


# %%
# initial downsampling so there isn't so much on the gpu
mindownJ = np.min(np.array(downJs),0)
xJ,J = emlddmm.downsample_image_domain(xJ,J,mindownJ)
W0 = emlddmm.downsample(W0,mindownJ)
downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
dJ = [x[1]-x[0] for x in xJ]
nJ = np.array(J.shape,dtype=int)
# update our config variable
config['downJ'] = downJs


# %%
imp.reload(emlddmm)
# visualize initial affine
if 'A' in config:
    A = np.array(config['A']).astype(float)
else:
    A = np.eye(4)
# this affine matrix should be 4x4, but it may be 1x4x4
if A.ndim > 2:
    A = A[0]
Ai = np.linalg.inv(A)
XJ = np.stack(np.meshgrid(*xJ,indexing='ij'),-1)
Xs = (Ai[:3,:3]@XJ[...,None])[...,0] + Ai[:3,-1]
out = emlddmm.interp(xI,I,Xs.transpose((3,0,1,2)))
fig = emlddmm.draw(out,xJ)
fig[0].suptitle('Initial transformed atlas')


# %%
print('shape I: ',I.shape)
print('shape J: ', J.shape)


# %%
imp.reload(emlddmm)
device = 'cuda:0'
#device = 'cpu'
output = emlddmm.emlddmm_multiscale(I=I,xI=[xI],J=J,xJ=[xJ],W0=W0,device=device,**config)


# %%
imp.reload(emlddmm)
emlddmm.write_transform_outputs(output_dir,output[-1])


# %%
# get labels
xS,S,title,names = emlddmm.read_data(label_name)


# %%
imp.reload(emlddmm)
emlddmm.write_qc_outputs(output_dir,output[-1],xI,I,xJ,J,xS=xS,S=S.astype(float))


# %%
# test it, forward transform is used for computing a transformed target
# that is, visualize the target deformed to match the atlas
imp.reload(emlddmm)
Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))
Xout = emlddmm.compose_sequence(output_dir,Xin)
Jt = emlddmm.apply_transform_float(xJ,J,Xout)


# %%
import matplotlib.pyplot as plt
emlddmm.draw(Jt,xI)
plt.gcf().suptitle('Transformed target')


# %%
# test it, backward transform is used for computing a transformed atlas
# that is, visualize the atlas deformed to match the target
Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xJ]))
Xout = emlddmm.compose_sequence(output_dir,Xin,direction='b')
It = emlddmm.apply_transform_float(xI,I,Xout)
St = emlddmm.apply_transform_int(xS,S,Xout)


# %%
St_ = St.float()
emlddmm.draw(It,xJ)
emlddmm.draw(torch.cat((St_%7,St_%5,St_%3)),xJ)


# %%
atlas_output_dir = os.path.join(output_dir,'to_atlas')
if not os.path.isdir(atlas_output_dir): os.mkdir(atlas_output_dir)
target_output_dir = os.path.join(output_dir,'to_target')
if not os.path.isdir(target_output_dir): os.mkdir(target_output_dir)


# %%
# write out
imp.reload(emlddmm)
emlddmm.write_data(os.path.join(atlas_output_dir,'target_to_atlas.vtk'),xI,Jt,'target_to_atlas')

emlddmm.write_data(os.path.join(target_output_dir,'atlas_to_target.vtk'),xI,It,'atlas_to_target')
emlddmm.write_data(os.path.join(target_output_dir,'atlas_seg_to_target.vtk'),xI,St,'atlas_seg_to_target')


# %%
# save config used to output
with open(config_file) as f:
    config = json.load(f)
with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config, f)


# %%
# TODO write out data at original resolution


# %%
# with open('test','wt') as f:
#     f.write()


# %%



# %%



