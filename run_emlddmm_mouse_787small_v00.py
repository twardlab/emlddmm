import numpy as np
import torch
import matplotlib.pyplot as plt
import emlddmm
import json
from datetime import datetime
import os

# todo list in this cell
# explicitly write out forward and inverse transforms instead of velocity field
# specifics for 3D to 2D mapping
# apply transforms to new data from command line
# different data types: either preprocess data and convert format, or modify code to support another data type
# reading and writing in other data types: especially nifty nii, I would rely on nibabel to do this

# maybe I want to use the exvivo, not the atlas. that can be a version 2.
atlas_name = 'Allen_Atlas_vtk/ara_nissl_50.vtk'
label_name = 'Allen_Atlas_vtk/annotation_50.vtk'
target_name = 'C:\\Users\\BGAdmin\\data\\MD816\\MD816_STIF'
config_file = 'config787small.json'
time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
output_dir = '787_small_test_outputs_'+time

with open(config_file) as f:
    config = json.load(f)
# I'm getting this for initial downsampling for preprocessing
downIs = config['downI']
downJs = config['downJ']

# atlas
xI,I,title,names = emlddmm.read_data(atlas_name)
I = I.astype(float)
# normalize
I /= np.mean(np.abs(I))
dI = np.array([x[1]-x[0] for x in xI])
print(dI)
fig = emlddmm.draw(I,xI)
# fig[0].suptitle('Atlas image')
# fig[0].canvas.draw()
# plt.show()

# initial downsampling so there isn't so much on the gpu
mindownI = np.min(np.array(downIs),0)
xI,I = emlddmm.downsample_image_domain(xI,I,mindownI)
downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
dI = [x[1]-x[0] for x in xI]
print(dI)
nI = np.array(I.shape,dtype=int)
# update our config variable
config['downI'] = downIs

# target
xJ,J,title,names = emlddmm.read_data(target_name, down=[1,4,4]) # try downsampling by 4
if 'mask' in names:
    maskind = names.index('mask')
    W0 = J[maskind]
    J = J[np.arange(J.shape[0])!=maskind]
dJ = np.array([x[1]-x[0] for x in xJ])
print(dJ)
J = J.astype(float)
fig = emlddmm.draw(J,xJ, vmin=0, vmax=1)
# fig[0].suptitle('Target image')
# fig[0].canvas.draw()
# print(J.shape)
# plt.show()

# initial downsampling so there isn't so much on the gpu
mindownJ = np.min(np.array(downJs),0)
xJ,J = emlddmm.downsample_image_domain(xJ,J,mindownJ)
W0 = emlddmm.downsample(W0,mindownJ)
downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
dJ = [x[1]-x[0] for x in xJ]
nJ = np.array(J.shape,dtype=int)
# update our config variable
config['downJ'] = downJs
# fig = emlddmm.draw(J,xJ)
# fig[0].suptitle('Initial downsampled target')
# fig[0].canvas.draw()
# plt.show()

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
# fig = emlddmm.draw(out,xJ)
# fig[0].suptitle('Initial transformed atlas')
# fig[0].canvas.draw()
# plt.show()

# list the config options
config['downI'] = [[2,2,2]] # make only one scale for memory purposes
for k in config:
    print(f'{k} : {config[k]}')

device = 'cuda:0'
#device = 'cpu'
output = emlddmm.emlddmm_multiscale(I=I,xI=[xI],J=J,xJ=[xJ],W0=W0,device=device,**config)

emlddmm.write_transform_outputs(output_dir,output[-1])

# get labels

xS,S,title,names = emlddmm.read_data(label_name,endian='l')
emlddmm.write_qc_outputs(output_dir,output[-1],xI,I,xJ,J,xS=xS,S=S.astype(float))

# save config used to output
with open(config_file) as f:
    config = json.load(f)
with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config, f)