import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample
import glob
import os
import nrrd
import json
import re
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN


# display
def draw(J,xJ=None,fig=None,n_slices=5,vmin=None,vmax=None):    
    if type(J) == torch.Tensor:
        J = J.detach().clone().cpu()
    J = np.array(J)
    if xJ is None:
        nJ = J.shape[-3:]
        xJ = [np.arange(n) - (n-1)/2.0 for n in nJ] 
    if type(xJ[0]) == torch.Tensor:
        xJ = [np.array(x.detach().clone().cpu()) for x in xJ]
    xJ = [np.array(x) for x in xJ]
    
    if fig is None:
        fig = plt.figure()
    fig.clf()    
    if vmin is None:
        vmin = np.quantile(J,0.001,axis=(-1,-2,-3))
    if vmax is None:
        vmax = np.quantile(J,0.999,axis=(-1,-2,-3))
    vmin = np.array(vmin)
    vmax = np.array(vmax)    
    # I will normalize data with vmin, and display in 0,1
    if vmin.ndim == 0:
        vmin = np.repeat(vmin,J.shape[0])
    if vmax.ndim == 0:
        vmax = np.repeat(vmax,J.shape[0])
    if len(vmax) >= 2 and len(vmin) >= 2:
        # for rgb I'll scale it, otherwise I won't, so I can use colorbars
        J -= vmin[:,None,None,None]
        J /= (vmax[:,None,None,None] - vmin[:,None,None,None])
        J[J<0] = 0
        J[J>1] = 1
        vmin = 0.0
        vmax = 1.0
    # I will only sohw the first 3 channels
    if J.shape[0]>3:
        J = J[:3]
    if J.shape[0]==2:
        J = np.stack((J[0],J[1],J[0]))
    
    
    axs = []
    axsi = []
    # ax0
    slices = np.round(np.linspace(0,J.shape[1],n_slices+2)[1:-1]).astype(int)    
    # for origin upper (default), extent is x (small to big), then y reversed (big to small)
    extent = (xJ[2][0],xJ[2][-1],xJ[1][-1],xJ[1][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1)
        ax.imshow(J[:,slices[i]].transpose(1,2,0).squeeze(),vmin=vmin,vmax=vmax,aspect='equal',extent=extent)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax1
    slices = np.round(np.linspace(0,J.shape[2],n_slices+2)[1:-1]).astype(int)    
    extent = (xJ[2][0],xJ[2][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1+n_slices)
        ax.imshow(J[:,:,slices[i]].transpose(1,2,0).squeeze(),vmin=vmin,vmax=vmax,aspect='equal',extent=extent)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax2
    slices = np.round(np.linspace(0,J.shape[3],n_slices+2)[1:-1]).astype(int)    
    extent = (xJ[1][0],xJ[1][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):        
        ax = fig.add_subplot(3,n_slices,i+1+n_slices*2)
        ax.imshow(J[:,:,:,slices[i]].transpose(1,2,0).squeeze(),vmin=vmin,vmax=vmax,aspect='equal',extent=extent)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    
    return fig,axs

# Common load function for vtk, nrrd volumes and microscopy slices
def load_data(fname):
    root, ext = os.path.splitext(fname)
    if ext == '':
        print('loading target images')
        fig, ax = plt.subplots()
        ax = [ax]
        # current limitation
        # requires the word 'present'
        # requires the first image to be present
        # expects data type to be in 0,1
        # assumes space directions are diagonal
        # todo: origin

        data = []
        # load the one tsv file
        tsv_name = os.path.join(fname, 'samples.tsv')
        with open(tsv_name, 'rt') as f:
            for count, line in enumerate(f):
                line = line.strip()
                key = '\t' if '\t' in line else '    '
                if count == 0:
                    headings = re.split(key, line)
                    continue
                data.append(re.split(key, line))
        data_ = np.zeros((len(data), len(data[0])), dtype=object)
        for i in range(data_.shape[0]):
            for j in range(data_.shape[1]):
                try:
                    data_[i, j] = data[i][j]
                except:
                    data_[i, j] = ''
        data = data_

        # now we will loop through the files and get the sizes
        nJ_ = np.zeros((data.shape[0], 3), dtype=int)
        J_ = []
        for i in range(data.shape[0]):
            namekey = data[i, 0]
            jsonfile = glob.glob(os.path.join(fname, '*' + namekey + '*.json'))
            present = data[i, 3] == 'present'
            if not present:
                if i == 0:
                    raise Exception('First image is not present')
                J_.append(np.array([[[0.0, 0.0, 0.0]]]))
                continue
            with open(jsonfile[0]) as f:
                jsondata = json.load(f)
            # nJ_[i] = np.array(jsondata['Sizes'])

            # this should contain an image and a json
            if 'DataFile' in jsondata:
                image_name = jsondata['DataFile']
            elif 'DataFIle' in jsondata:
                image_name = jsondata['DataFIle']

            J__ = plt.imread(os.path.join(fname, image_name))
            if J__.dtype == np.uint8:
                J__ = J__.astype(float) / 255.0

            J__ = J__[..., :3]  # no alpha
            nJ_[i] = np.array(J__.shape)

            J_.append(J__)

            if not i % 20:
                ax[0].cla()
                ax[0].imshow(J__)

                fig.suptitle(f'slice {i} of {data.shape[0]}: {image_name}')
                fig.canvas.draw()

            # the domain
            if i == 0:
                dJ = np.diag(np.array(jsondata['SpaceDirections'][1:]))[::-1]

            # note the order needs to be reversed

            # if this is the first file we want to set up a 3D volume

        nJm = np.max(nJ_, 0)
        nJm = (np.quantile(nJ_, 0.99, axis=0) * 1.1).astype(int)
        # this will look for outliers when there are a small number,
        # really there just shouldn't be outliers
        nJsave = np.copy(nJ_)

        print('padding and assembling into 3D volume')
        J = np.zeros((len(data), nJm[0], nJm[1], 3))
        W0 = np.zeros((len(data), nJm[0], nJm[1]))
        for i in range(len(J_)):
            J__ = J_[i]
            topad = nJm - np.array(nJ_[i])
            # just pad on the left and I'll fill it in
            topad = np.array(((topad[0] // 2, 0), (topad[1] // 2, 0), (0, 0)))
            # if there are any negative values I need to crop, I'll just crop on the right
            if np.any(np.array(topad) < 0):
                if topad[0][0] < 0:
                    J__ = J__[:J.shape[1]]
                    topad[0][0] = 0
                if topad[1][0] < 0:
                    J__ = J__[:, :J.shape[2]]
                    topad[1][0] = 0
            Jp = np.pad(J__, topad, constant_values=np.nan)
            W0_ = np.logical_not(np.isnan(Jp[..., 0]))
            Jp[np.isnan(Jp)] = 0
            W0[i, :W0_.shape[0], :W0_.shape[1]] = W0_
            J[i, :W0_.shape[0], :W0_.shape[1], :] = Jp
        J = np.transpose(J, (3, 0, 1, 2))
        nJ = np.array(J.shape)
        xJ = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nJ[1:], dJ)]
        W0 = W0 * np.logical_not(np.all(J == 0.0, 0))
        W0 = W0 * np.logical_not(np.all(J == 1.0, 0))
        # free up memory, not sure if this is necessary inside a function
        del J_
        dJ, nJ = []

    elif ext == '.vtk':
        reader = vtkStructuredPointsReader()
        reader.SetFileName(fname)
        reader.Update()
        data = reader.GetOutput()
        vtk_data = data.GetPointData().GetScalars()
        J = VN.vtk_to_numpy(vtk_data).astype(float)
        nJ = np.concatenate((np.array(data.GetDimensions()), np.array([1])))
        J = J.reshape(nJ[2], nJ[1], nJ[0], nJ[3])
        J = J.transpose(3, 0, 2, 1)
        dJ = np.array(data.GetSpacing())
        xJ = [np.arange(float(n)) * d - (n - 1) * d / 2.0 for n, d in zip(nJ, dJ)]
        W0 = []
    elif ext == '.nrrd':
        J, hdr = nrrd.read(fname)
        J = J.astype(float)
        J = J ** 0.5
        J /= np.sqrt(np.mean(J ** 2))
        J = J[None]
        dJ = np.diag(hdr['space directions']).astype(float)
        nJ = J.shape
        xJ = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nJ[1:], dJ)]
        W0 = []

    return dJ, nJ, xJ, J, W0
    
# def load_slices(target_name):
#     print('loading target images')
#     fig,ax = plt.subplots()
#     ax = [ax]
#     # current limitation
#     # requires the word 'present'
#     # requires the first image to be present
#     # expects data type to be in 0,1
#     # assumes space directions are diagonal
#     # todo: origin
#
#     data = []
#     # load the one tsv file
#     tsv_name = os.path.join(target_name, 'samples.tsv' )
#     with open(tsv_name,'rt') as f:
#         for count,line in enumerate(f):
#             line = line.strip()
#             key = '\t' if '\t' in line else '    '
#             if count == 0:
#                 headings = re.split(key,line)
#                 continue
#             data.append(re.split(key,line))
#     data_ = np.zeros((len(data),len(data[0])),dtype=object)
#     for i in range(data_.shape[0]):
#         for j in range(data_.shape[1]):
#             try:
#                 data_[i,j] = data[i][j]
#             except:
#                 data_[i,j] = ''
#     data = data_
#
#     # now we will loop through the files and get the sizes
#     nJ_ = np.zeros((data.shape[0],3),dtype=int)
#     J_ = []
#     for i in range(data.shape[0]):
#         namekey = data[i,0]
#         jsonfile = glob.glob(os.path.join(target_name,'*'+namekey+'*.json'))
#         present = data[i,3] == 'present'
#         if not present:
#             if i == 0:
#                 raise Exception('First image is not present')
#             J_.append(np.array([[[0.0,0.0,0.0]]]))
#             continue
#         with open(jsonfile[0]) as f:
#             jsondata = json.load(f)
#         #nJ_[i] = np.array(jsondata['Sizes'])
#
#
#         # this should contain an image and a json
#         if 'DataFile' in jsondata:
#             image_name = jsondata['DataFile']
#         elif 'DataFIle' in jsondata:
#             image_name = jsondata['DataFIle']
#
#         J__ = plt.imread(os.path.join(target_name,image_name))
#         if J__.dtype == np.uint8:
#             J__ = J__.astype(float)/255.0
#
#         J__ = J__[...,:3] # no alpha
#         nJ_[i] = np.array(J__.shape)
#
#         J_.append(J__)
#
#         if not i%20:
#             ax[0].cla()
#             ax[0].imshow(J__)
#
#
#             fig.suptitle(f'slice {i} of {data.shape[0]}: {image_name}')
#             fig.canvas.draw()
#
#         # the domain
#         if i == 0:
#             dJ = np.diag(np.array(jsondata['SpaceDirections'][1:]))[::-1]
#
#         # note the order needs to be reversed
#
#         # if this is the first file we want to set up a 3D volume
#
#     nJm = np.max(nJ_,0)
#     nJm = (np.quantile(nJ_,0.99,axis=0)*1.1).astype(int)
#     # this will look for outliers when there are a small number,
#     # really there just shouldn't be outliers
#     nJsave = np.copy(nJ_)
#
#     print('padding and assembling into 3D volume')
#     J = np.zeros((len(data),nJm[0],nJm[1],3))
#     W0 = np.zeros((len(data),nJm[0],nJm[1]))
#     for i in range(len(J_)):
#         J__ = J_[i]
#         topad = nJm - np.array(nJ_[i])
#         # just pad on the left and I'll fill it in
#         topad = np.array(((topad[0]//2,0),(topad[1]//2,0),(0,0)))
#         # if there are any negative values I need to crop, I'll just crop on the right
#         if np.any(np.array(topad)<0):
#             if topad[0][0] < 0:
#                 J__ = J__[:J.shape[1]]
#                 topad[0][0] = 0
#             if topad[1][0] < 0:
#                 J__ = J__[:,:J.shape[2]]
#                 topad[1][0] = 0
#         Jp = np.pad(J__,topad,constant_values=np.nan)
#         W0_ = np.logical_not(np.isnan(Jp[...,0]))
#         Jp[np.isnan(Jp)] = 0
#         W0[i,:W0_.shape[0],:W0_.shape[1]] = W0_
#         J[i,:W0_.shape[0],:W0_.shape[1],:] = Jp
#     J = np.transpose(J,(3,0,1,2))
#     nJ = np.array(J.shape)
#     xJ = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(nJ[1:],dJ)]
#     W0 = W0 * np.logical_not(np.all(J==0.0,0))
#     W0 = W0 * np.logical_not(np.all(J==1.0,0))
#     # free up memory, not sure if this is necessary inside a function
#     del J_
#     return xJ,J,W0
    

            
# resampling
def sinc_resample_numpy(I,n):
    ''' This function does sinc resampling using numpy rfft
    torch does not let us control behavior of fft well enough
    This is intended to be used to resample velocity fields if necessary
    I am realy only intending it to be used for upsampling
    '''
    Id = np.array(I)
    for i in range(len(n)):        
        if I.shape[i] == n[i]:
            continue
        Id = np.fft.irfft(np.fft.rfft(Id,axis=i),axis=i,n=n[i])
    # output with correct normalization
    return Id*np.prod(Id.shape)/np.prod(I.shape) 
    

def downsample_ax(I,down,ax):
    nd = list(I.shape)        
    nd[ax] = nd[ax]//down
    if type(I) == torch.Tensor:
        Id = torch.zeros(nd,device=I.device,dtype=I.dtype)
    else:
        Id = np.zeros(nd,dtype=I.dtype)
    for d in range(down):
        if ax==0:        
            Id += I[d:down*nd[ax]:down]
        elif ax==1:        
            Id += I[:,d:down*nd[ax]:down]
        elif ax==2:
            Id += I[:,:,d:down*nd[ax]:down]
        elif ax==3:
            Id += I[:,:,:,d:down*nd[ax]:down]
        elif ax==4:
            Id += I[:,:,:,:,d:down*nd[ax]:down]
        elif ax==5:
            Id += I[:,:,:,:,:,d:down*nd[ax]:down]
        # ... should be enough but there really has to be a better way to do this        
    return Id/down
def downsample(I,down):
    down = list(down)
    while len(down) < len(I.shape):
        down.insert(0,1)    
    if type(I) == torch.Tensor:
        Id = torch.clone(I)
    else:
        Id = np.copy(I)
    for i,d in enumerate(down):
        if d==1:
            continue
        Id = downsample_ax(Id,d,i)
    return Id
def downsample_image_domain(xI,I,down): 
    if len(xI) != len(down):
        raise Exception('Length of down and xI must be equal')
    Id = downsample(I,down)    
    xId = []
    for i,d in enumerate(down):
        xId.append(downsample_ax(xI[i],d,0))
    return xId,Id

# build an interp function from grid sample
def interp(x,I,phii,**kwargs):
    '''
    Interpolate the image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D arrays with first channel storing component)
    '''
    # first we have to normalize phii to the range -1,1    
    phii = torch.clone(phii)
    for i in range(3):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done
        
    # NOTE I should check that I can reproduce identity
    # note that phii must now store x,y,z along last axis
    # is this the right order?
    # I need to put batch (none) along first axis
    # what order do the other 3 need to be in?    
    out = grid_sample(I[None],phii.flip(0).permute((1,2,3,0))[None],align_corners=True,**kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension
    
    return out[0]
    
# now we need to create a flow
# timesteps will be along the first axis
def v_to_phii(xv,v):
    XV = torch.stack(torch.meshgrid(xv))
    phii = torch.clone(XV)
    dt = 1.0/v.shape[0]
    for t in range(v.shape[0]):
        Xs = XV - v[t]*dt
        phii = interp(xv,phii-XV,Xs)+Xs
    return phii
        
def emlddmm(**kwargs):
    # required arguments are
    # I - atlas image, size C x slice x row x column
    # xI - list of pixels locations in I, corresponding to each axis other than channels
    # J - target image, size C x slice x row x column
    # xJ - list of pixel locations in J
    # other parameters are specified in a dictionary with defaults listed below
    # if you provide an input for PARAMETER it will be used as an initial guess, 
    # unless you specify update_PARAMETER=False, in which case it will be fixed
    # if you specify update_sigmaM=False it will use sum of square error, 
    # otherwise it will use log determinant of diagonal covariance matrix
    
    # I should move them to torch and put them on the right device
    I = kwargs['I']
    J = kwargs['J']
    xI = kwargs['xI']
    xJ = kwargs['xJ']
    
    ##########################################################################################################
    # everything else is optinal, defaults are below
    defaults = {'nt':5,
                'eA':1e5,
                'ev':2e3,
                'order':1, # order of polynomial
                'n_draw':10,
                'sigmaR':1e6,
                'n_iter':2000,
                'n_e_step':10,
                'v_start':200,
                'n_reduce_step':10,
                'v_expand_factor':0.2,
                'v_res_factor':2.0,
                'a':None, # note default below dv[0]*2.0
                'p':2.0,    
                'aprefactor':0.1, # in terms of voxels in the downsampled atlas
                'device':None, # cuda:0 if available otherwise cpu
                'dtype':torch.float,
                'downI':[1,1,1],
                'downJ':[1,1,1],      
                'W0':None,
                'priors':None,
                'update_priors':True,
                'full_outputs':False,
                'muB':None,
                'update_muB':True,
                'muA':None,
                'update_muA':True,
                'sigmaA':None,
                'update_sigmaA':True,
                'sigmaB':None,
                'update_sigmaB':True,
                'sigmaM':None,
                'update_sigmaM':True,
                'A':None,
                'v':None,
                'A2d':None,     
                'eA2d':1e-3, # removed eL and eT using metric, need to double check 2d case works well
                'slice_matching':False, # if true include rigid motions and contrast on each slice
                'slice_matching_start':0,
               }
    defaults.update(kwargs)
    kwargs = defaults
    device = kwargs['device']
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    dtype = kwargs['dtype']
    if dtype is None:
        dtype = torch.float
    
    # move the above to the right device
    I = torch.as_tensor(I,device=device,dtype=dtype)
    J = torch.as_tensor(J,device=device,dtype=dtype)
    xI = [torch.as_tensor(x,device=device,dtype=dtype) for x in xI]
    xJ = [torch.as_tensor(x,device=device,dtype=dtype) for x in xJ]
    
    
    ##########################################################################################################
    nt = kwargs['nt']
    eA = kwargs['eA']
    ev = kwargs['ev']
    eA2d = kwargs['eA2d']
    
    order = kwargs['order'] 
    
    sigmaR = kwargs['sigmaR']
    n_iter = kwargs['n_iter']
    v_start = kwargs['v_start']
    n_draw = kwargs['n_draw']
    n_e_step = kwargs['n_e_step']
    n_reduce_step = kwargs['n_reduce_step']
    
    v_expand_factor = kwargs['v_expand_factor']
    v_res_factor = kwargs['v_res_factor']
    
    a = kwargs['a']
    p = kwargs['p']
    aprefactor = kwargs['aprefactor']
        
    
    downI = kwargs['downI']
    downJ = kwargs['downJ']
    W0 = kwargs['W0']
    if W0 is None:
        W0 = torch.ones_like(J)
    else:
        W0 = torch.as_tensor(W0,device=device,dtype=dtype)
    N = torch.sum(W0) 
    priors = kwargs['priors']
    update_priors = kwargs['update_priors']
    if priors is None and not update_priors:        
        priors = [1.0,1.0,1.0]
    full_outputs = kwargs['full_outputs']
    muA = kwargs['muA']
    update_muA = kwargs['update_muA']
    muB = kwargs['muB']
    update_muB = kwargs['update_muB']
    sigmaA = kwargs['sigmaA']
    update_sigmaA = kwargs['update_sigmaA']
    sigmaB = kwargs['sigmaB']
    update_sigmaB = kwargs['update_sigmaB']
    sigmaM = kwargs['sigmaM']
    update_sigmaM = kwargs['update_sigmaM']
    
    
    A = kwargs['A']    
    v = kwargs['v']
    slice_matching = kwargs['slice_matching']
    if slice_matching is None:
        slice_matching = False
    if slice_matching:
        A2d = kwargs['A2d']
    slice_matching_start = kwargs['slice_matching_start']
    
    
    
    ##########################################################################################################    
    # domain
    dI = torch.tensor([xi[1]-xi[0] for xi in xI],device=device,dtype=dtype)
    dJ = torch.tensor([xi[1]-xi[0] for xi in xJ],device=device,dtype=dtype)    
    nI = torch.tensor(I.shape,dtype=dtype,device=device)
    nJ = torch.tensor(J.shape,dtype=dtype,device=device)
    
    # set up a domain for xv    
    # I'll put it a bit bigger than xi
    dv = dI*v_res_factor # we want this to be independent of the I downsampling sampling
    print(f'dv {dv}')
    if a is None:
        a = dv[0]*2.0 # so this is also independent of the I downsampling amount
    print(f'a scale is {a}')
    x0v = [x[0] - (x[-1]-x[0])*v_expand_factor for x in xI]
    x1v = [x[-1] + (x[-1]-x[0])*v_expand_factor for x in xI]
    xv = [torch.arange(x0,x1,d,device=device,dtype=dtype) for x0,x1,d in zip(x0v,x1v,dv)]
    nv = torch.tensor([len(x) for x in xv],device=device,dtype=dtype)
    XV = torch.stack(torch.meshgrid(xv))
    #print(f'velocity size is {nv}')
    
    
    # downample    
    xI,I = downsample_image_domain(xI,I,downI)
    xJ,J = downsample_image_domain(xJ,J,downJ)
    dI *= torch.prod(torch.tensor(downI,device=device,dtype=dtype))
    dJ *= torch.prod(torch.tensor(downJ,device=device,dtype=dtype))
    vminI = [np.quantile(J_.cpu().numpy(),0.001) for J_ in I]
    vmaxI = [np.quantile(J_.cpu().numpy(),0.999) for J_ in I]
    vminJ = [np.quantile(J_.cpu().numpy(),0.001) for J_ in J]
    vmaxJ = [np.quantile(J_.cpu().numpy(),0.999) for J_ in J]
    
    
    W0 = downsample(W0,downJ)
    XI = torch.stack(torch.meshgrid(xI))
    XJ = torch.stack(torch.meshgrid(xJ))
    
    
    # build an affine metric for 3D affinee
    # this one is based on pullback metric from action on voxel locations (not image)
    # build a basis in lexicographic order and push forward using the voxel locations
    XI_ = XI.permute((1,2,3,0))[...,None]    
    E = []
    for i in range(3):
        for j in range(4):
            E.append(   ((torch.arange(4,dtype=dtype,device=device)[None,:] == j)*(torch.arange(4,dtype=dtype,device=device)[:,None] == i))*1.0 )
    # it would be nice to define scaling so that the norm of a perturbation had units of microns
    # e.g. root mean square displacement
    g = torch.zeros((12,12),dtype=dtype,device=device)
    for i in range(len(E)):
        EiX = (E[i][:3,:3]@XI_)[...,0] + E[i][:3,-1]
        for j in range(len(E)):
            EjX = (E[j][:3,:3]@XI_)[...,0] + E[j][:3,-1]
            # matrix multiplication            
            g[i,j] = torch.sum(EiX*EjX) #* torch.prod(dI) # because gradient has a factor of N in it, I think its a good idea to do sum
    gi = torch.inverse(g)           

    # TODO affine metric for 2D affine
    # I'll use a quick hack for now
    # this is again based on pullback metric for voxel locations
    # need to verify that this is correct given possibly moving coordinate
    # maybe better 
    E = []
    
    for i in range(2):
        for j in range(3):
            E.append(   ((torch.arange(3,dtype=dtype,device=device)[None,:] == j)*(torch.arange(3,dtype=dtype,device=device)[:,None] == i))*1.0 )
    g2d = torch.zeros((6,6),dtype=dtype,device=device)
    
    for i in range(len(E)):
        EiX = (E[i][:2,:2]@XI_[0,...,1:,:])[...,0] + E[i][:2,-1]
        for j in range(len(E)):
            EjX = (E[j][:2,:2]@XI_[0,...,1:,:])[...,0] + E[j][:2,-1]
            g2d[i,j] = torch.sum(EiX*EjX)
    g2di = torch.inverse(g2d)
            
    # build energy operator for velocity
    fv = [torch.arange(n,device=device,dtype=dtype)/d/n for n,d in zip(nv,dv)]
    FV = torch.stack(torch.meshgrid(fv))

    LL = (1.0 - 2.0*a**2 * 
              ( (torch.cos(2.0*np.pi*FV[0]*dv[0]) - 1)/dv[0]**2  
            + (torch.cos(2.0*np.pi*FV[1]*dv[1]) - 1)/dv[1]**2  
            + (torch.cos(2.0*np.pi*FV[2]*dv[2]) - 1)/dv[2]**2   ) )**(p*2)
    K = 1.0/LL

    LLpre = (1.0 - 2.0*(aprefactor*torch.max(dI))**2 * 
             ( (torch.cos(2.0*np.pi*FV[0]*dv[0]) - 1)/dv[0]**2  
             + (torch.cos(2.0*np.pi*FV[1]*dv[1]) - 1)/dv[1]**2  
             + (torch.cos(2.0*np.pi*FV[2]*dv[2]) - 1)/dv[2]**2   ) )**(p*2)
    Kpre = 1.0/LLpre
    KK = K*Kpre

    
    
    
    # now initialize variables and optimizers    
    vsize = (nt,3,int(nv[0]),int(nv[1]),int(nv[2]))
    
    if v is None:
        v = torch.zeros(vsize,dtype=dtype,device=device,requires_grad=True)
    else:
        # check the size
        if torch.all(torch.as_tensor(v.shape,device=device,dtype=dtype)==torch.as_tensor(vsize,device=device,dtype=dtype)):
            v = torch.as_tensor(v,device=device,dtype=dtype) 
            v.requires_grad = True
        else:
            if v[1] != vsize[1]:
                raise Exception('Initial velocity must have 3 components')
            # resample it
            v = sinc_resample_numpy(v,vsize)
            v = torch.as_tensor(v,device=device,dtype=dtype)
            v.requires_grad = True
            
            
    if A is None:
        A = torch.eye(4,requires_grad=True, device=device, dtype=dtype)
    else:
        A = torch.as_tensor(A,device=device,dtype=dtype)
        A.requires_grad = True
        
    if slice_matching:
        if A2d is None:
            A2d = torch.eye(3,device=device,dtype=dtype)[None].repeat(J.shape[1],1,1)
            A2d.requires_grad = True
        else:
            A2d = torch.as_tensor(A2d, device=device, dtype=dtype,requires_grad=True)
        # if slice matching is on we want to add xy translation in A to A2d
        with torch.no_grad():
            A2di = torch.inverse(A2d)
            vec = A[1:3,-1]
            A2d[:,:2,-1] += (A2di[:,:2,:2]@vec[...,None])[...,0]
            A[1:3,-1] = 0
    
    WM = torch.ones(J.shape[1:], device=device, dtype=dtype)*0.8
    WA = torch.ones(J.shape[1:], device=device, dtype=dtype)*0.1
    WB = torch.ones(J.shape[1:], device=device, dtype=dtype)*0.1

    if muA is None:         
        #muA = torch.tensor([torch.max(J_[W0>0]) for J_ in J],dtype=dtype,device=device)      
        muA = torch.tensor([torch.tensor(np.quantile( (J_[W0>0]).cpu().numpy(), 0.999 ),dtype=dtype,device=device) for J_ in J],dtype=dtype,device=device)
    else: # if we input some value, we'll just use that
        muA = torch.tensor(muA,dtype=dtype,device=device)    
    if muB is None:
        #muB = torch.tensor([torch.min(J_[W0>0]) for J_ in J],dtype=dtype,device=device)  
        muB = torch.tensor([torch.tensor(np.quantile( (J_[W0>0]).cpu().numpy(), 0.001 ),dtype=dtype,device=device) for J_ in J],dtype=dtype,device=device)
    else: # if we input a value we'll just use that
        muB = torch.tensor(muB,dtype=dtype,device=device)        
    # TODO update to covariance, for now just diagonal
    DJ = torch.prod(dJ)
    if sigmaM is None:
        sigmaM = torch.std(J,dim=(1,2,3))*1.0#*DJ
    else:
        sigmaM = torch.as_tensor(sigmaM,device=device,dtype=dtype)
    if sigmaA is None:
        sigmaA = torch.std(J,dim=(1,2,3))*5.0#*DJ
    else:
        sigmaA = torch.as_tensor(sigmaA,device=device,dtype=dtype)
    if sigmaB is None:
        sigmaB = torch.std(J,dim=(1,2,3))*2.0#*DJ
    else:
        sigmaB = torch.as_tensor(sigmaB,device=device,dtype=dtype)
    
    figE,axE = plt.subplots(1,3)
    figA,axA = plt.subplots(2,2)
    axA = axA.ravel()
    if slice_matching:
        figA2d,axA2d = plt.subplots(2,2)
        axA2d = axA2d.ravel()
    figI = plt.figure()
    figfI = plt.figure()
    figErr = plt.figure()
    figJ = plt.figure()
    figV = plt.figure()
    figW = plt.figure()


    Esave = []
    Lsave = []
    Tsave = []
    if slice_matching:
        T2dsave = []
        L2dsave = []
    maxvsave = []
    sigmaMsave = []
    sigmaAsave = []
    sigmaBsave = []

    
    ################################################################################
    # end of setup, start optimization loop
    for it in range(n_iter):
        # get the transforms        
        phii = v_to_phii(xv,v) # on the second iteration I'm getting an error here 
        Ai = torch.inverse(A)
        # 2D transforms
        if slice_matching:
            A2di = torch.inverse(A2d)            
            XJ_ = torch.clone(XJ)
            # leave z component the same (x0) and transform others                   
            XJ_[1:] = ((A2di[:,None,None,:2,:2]@ (XJ[1:].permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)            
        else:
            XJ_ = XJ

        # sample points for affine
        Xs = ((Ai[:3,:3]@XJ_.permute((1,2,3,0))[...,None])[...,0] + Ai[:3,-1]).permute((3,0,1,2))
        # for diffeomorphism
        phiiAi = interp(xv,phii-XV,Xs) + Xs
        
        # transform image
        AphiI = interp(xI,I,phiiAi)                

        # transform contrast (here assume atlas is dim 1)
        Nvoxels = AphiI.shape[0]*AphiI.shape[1]*AphiI.shape[2]*AphiI.shape[3] # why did I write 0, well its equal to 1 here
        B_ = torch.ones((Nvoxels,order+1),dtype=dtype,device=device)
        for i in range(order):
            B_[:,i+1] = AphiI.reshape(-1)**(i+1) # this assumes atlas is only dim 1, okay for now
        if not slice_matching:
            with torch.no_grad():
                # multiply by weight
                B__ = B_*(WM*W0).reshape(-1)[...,None]
                coeffs = torch.inverse(B__.T@B_) @ (B__.T@(J.reshape(J.shape[0],-1).T))                                    
            fAphiI = ((B_@coeffs).T).reshape(J.shape) # there are unnecessary transposes here, probably slowing down, to fix later
        else: # with slice matching I need to solve these equation for every slice
            with torch.no_grad():
                # multiply by weight
                B__ = B_*(WM*W0).reshape(-1)[...,None]
                coeffs = torch.inverse(B__.T@B_) @ (B__.T@(J.reshape(J.shape[0],-1).T))                                    
            fAphiI = ((B_@coeffs).T).reshape(J.shape) 
        
        err = (fAphiI - J)
        err2 = (err**2*(WM*W0))
        # most of my updates are below (at the very end), but I'll update this here because it is in my cost function
        # note that in my derivation, the sigmaM should have this factor of DJ
        sseM = torch.sum( err2,(-1,-2,-3))
        if update_sigmaM:
            sigmaM = torch.sqrt(sseM/torch.sum(WM*W0))#*DJ
        
        sigmaMsave.append(sigmaM.detach().clone().cpu().numpy())
        JmmuA2 = (J - muA[...,None,None,None])**2
        JmmuB2 = (J - muB[...,None,None,None])**2
                
        if not it%n_e_step:
            with torch.no_grad():                                
                if update_priors:
                    priors = [torch.sum(WM*W0)/N, torch.sum(WA*W0)/N, torch.sum(WB*W0)/N]                         
                
                WM = (1.0/2.0/np.pi)**(J.shape[0]/2)/torch.prod(sigmaM) * torch.exp(  -torch.sum(err**2/2.0/sigmaM[...,None,None,None]**2,0)  )*priors[0]
                WA = (1.0/2.0/np.pi)**(J.shape[0]/2)/torch.prod(sigmaA) * torch.exp(  -torch.sum(JmmuA2/2.0/sigmaA[...,None,None,None]**2,0)  )*priors[1]
                WB = (1.0/2.0/np.pi)**(J.shape[0]/2)/torch.prod(sigmaB) * torch.exp(  -torch.sum(JmmuB2/2.0/sigmaB[...,None,None,None]**2,0)  )*priors[2]
                WS = WM+WA+WB                
                WS += torch.max(WS)*1e-12 # for numerical stability, but this may be leading to issues
                WM /= WS
                WA /= WS
                WB /= WS
                # todo think gabout MAP EM instead of ML EM, some dirichlet prior                
                # note, I seem to be getting black in my mask, why would it be black?
                



        # matching cost        
        # note the N here is the number of pixels (actually the sum of W0)
        if update_sigmaM:
            # log det sigma, when it is unknown
            EM = 0.5*N*priors[0]*torch.sum(torch.log(sigmaM))
            # Note in my derivation sigmaM is defined with a factor of prod dJ at the end
            # but everywhere it is used, except here, it gets divided by that factor
            # here including the factor just becomes another term in the sum, so I don't think I actually need it anywhere
        else:
            # sum of squares when it is known (note sseM includes weights)
            EM = torch.sum(sseM/sigmaM**2)*DJ/2.0
            # here I have a DJ
        
        
        # reg cost (note that with no complex, there are two elements on the last axis)
        vhat = torch.rfft(v,3,onesided=False)
        ER = torch.sum(torch.sum(vhat**2,(0,1,-1))*LL)/torch.prod(nv)*torch.prod(dv)/nt/2.0/sigmaR**2

        # total cost 
        E = EM + ER
        

        # gradient
        E.backward()

        # covector to vector
        vgrad = torch.irfft(torch.rfft(v.grad,3,onesided=False)*(KK)[None,None,...,None],3,onesided=False)
        Agrad = (gi@(A.grad[:3,:4].reshape(-1))).reshape(3,4)
        if slice_matching:
            A2dgrad = (g2di@(A2d.grad[:,:2,:3].reshape(A2d.shape[0],6,1))).reshape(A2d.shape[0],2,3)
            

        # plotting
        Esave.append([E.detach().cpu(),EM.detach().cpu(),ER.detach().cpu()])        
        Tsave.append(A[:3,-1].detach().clone().squeeze().cpu().numpy())
        Lsave.append(A[:3,:3].detach().clone().squeeze().reshape(-1).cpu().numpy())
        maxvsave.append(torch.max(torch.abs(v.detach())).clone().cpu().numpy())
        if slice_matching:
            T2dsave.append(A2d[:,:2,-1].detach().clone().squeeze().reshape(-1).cpu().numpy())
            L2dsave.append(A2d[:,:2,:2].detach().clone().squeeze().reshape(-1).cpu().numpy())
        # a nice check on step size would be to see if these are oscilating or monotonic
        if it > 10 and not it%n_reduce_step:
            reduce_factor = 0.9
            checksign0 = np.sign(maxvsave[-1] - maxvsave[-2])
            checksign1 = np.sign(maxvsave[-2] - maxvsave[-3])
            checksign2 = np.sign(maxvsave[-3] - maxvsave[-4])
            if np.any((checksign0 != checksign1)*(checksign1 != checksign2) ):
                ev *= reduce_factor
                print(f'Iteration {it} reducing ev to {ev}')

            checksign0 = np.sign(Tsave[-1] - Tsave[-2])
            checksign1 = np.sign(Tsave[-2] - Tsave[-3])
            checksign2 = np.sign(Tsave[-3] - Tsave[-4])
            reducedA = False
            if np.any((checksign0 != checksign1)*(checksign1 != checksign2)):
                eA *= reduce_factor
                print(f'Iteration {it}, translation oscilating, reducing eA to {eA}')
                reducedA = True
            checksign0 = np.sign(Lsave[-1] - Lsave[-2])
            checksign1 = np.sign(Lsave[-2] - Lsave[-3])
            checksign2 = np.sign(Lsave[-3] - Lsave[-4])
            if np.any( (checksign0 != checksign1)*(checksign1 != checksign2) ) and not reducedA:
                eA *= reduce_factor
                print(f'Iteration {it}, linear oscilating, reducing eA to {eA}')
                
            # to do, check sign for a2d


        if not it%n_draw:
            axE[0].cla()
            axE[0].plot(np.array(Esave)[:,0])
            axE[0].plot(np.array(Esave)[:,1])
            #axE[0].plot(np.array(Esave)[:,2])
            axE[0].set_title('Energy')
            axE[1].cla()
            axE[1].plot(np.array(Esave)[:,1])
            axE[1].set_title('Matching')
            axE[2].cla()
            axE[2].plot(np.array(Esave)[:,2])
            axE[2].set_title('Reg')


            _ = draw(AphiI.detach().cpu(),xJ,fig=figI,vmin=vminI,vmax=vmaxI)
            figI.suptitle('AphiI')
            _ = draw(fAphiI.detach().cpu(),xJ,fig=figfI,vmin=vminJ,vmax=vmaxJ)
            figfI.suptitle('fAphiI')
            _ = draw(fAphiI.detach().cpu() - J.cpu(),xJ,fig=figErr)
            figErr.suptitle('Err')
            _ = draw(J.cpu(),xJ,fig=figJ,vmin=vminJ,vmax=vmaxJ)
            figJ.suptitle('J')

            axA[0].cla()
            axA[0].plot(np.array(Tsave))
            axA[0].set_title('T')
            axA[1].cla()
            axA[1].plot(np.array(Lsave))
            axA[1].set_title('L')
            axA[2].cla()
            axA[2].plot(np.array(maxvsave))
            axA[2].set_title('maxv')
            axA[3].cla()
            axA[3].plot(sigmaMsave)
            axA[3].plot(sigmaAsave)
            axA[3].plot(sigmaBsave)
            axA[3].set_title('sigma')

            if slice_matching:
                axA2d[0].cla()
                axA2d[0].plot(np.array(T2dsave))
                axA2d[0].set_title('T2d')
                axA2d[1].cla()
                axA2d[1].plot(np.array(L2dsave))
                axA2d[1].set_title('L2d')
                
            
            figV.clf()
            draw(v[0].detach(),xv,fig=figV)
            figV.suptitle('velocity')

            figW.clf()
            draw(torch.stack((WM*W0,WA*W0,WB*W0)),xJ,fig=figW,vmin=0.0,vmax=1.0)
            figW.suptitle('Weights')

            figE.canvas.draw()
            figI.canvas.draw()
            figfI.canvas.draw()
            figErr.canvas.draw()
            figA.canvas.draw()
            figV.canvas.draw()
            figW.canvas.draw()
            figJ.canvas.draw()
            if slice_matching:
                figA2d.canvas.draw()


        # update
        with torch.no_grad():
            if it >= v_start:
                v -= vgrad*ev
            v.grad.zero_()
            
            A[:3] -= Agrad*eA
            A.grad.zero_()
            
            if slice_matching:
                
                # project A to isotropic and normal up
                #
                # TODO normal
                # todo figure out why this is not working
                # what I really should do is parameterize this group and work out a metric
                # I think its not working because of the origin for some reason
                u,s,v_ = torch.svd(A[:3,:3])
                #s = torch.diag(s)
                s = torch.exp(torch.mean(torch.log(s)))*torch.eye(3,device=device,dtype=dtype)
                A[:3,:3] = u@s@v_.T
                

                
                if it > slice_matching_start:
                    A2d[:,:2,:3] -= A2dgrad*eA2d # already scaled
                # project onto rigid
                u,s,v_ = torch.svd(A2d[:,:2,:2])
                A2d[:,:2,:2] = u@v_.transpose(1,2)
                A2d.grad.zero_()
                
                # move any xy translation into 2d
                # to do this I will have to account for any linear transformation
                vec = A[1:3,-1]
                A2d[:,:2,-1] += (A2di[:,:2,:2]@vec[...,None])[...,0]
                A[1:3,-1] = 0
            

            # other terms in m step, these don't actually matter until I update
            WAW0 = (WA*W0)[None]
            WAW0s = torch.sum(WAW0)
            if update_muA:
                muA = torch.sum(J*WAW0,dim=(-1,-2,-3))/WAW0s
            if update_sigmaA:
                sigmaA = torch.sqrt(  torch.sum(JmmuA2*WAW0,dim=(-1,-2,-3))/WAW0s  ) 
            sigmaAsave.append(sigmaA.detach().clone().cpu().numpy())

            WBW0 = (WB*W0)[None]
            WBW0s = torch.sum(WBW0)
            if update_muB:
                muB = torch.sum(J*WBW0,dim=(-1,-2,-3))/WBW0s
            if update_sigmaB:
                sigmaB = torch.sqrt(torch.sum(JmmuB2*WBW0,dim=(-1,-2,-3))/WBW0s)
            sigmaBsave.append(sigmaB.detach().clone().cpu().numpy())

        if not it%10:
            # todo print othefr info
            print(f'Finished iteration {it}')


            
    # outputs
    out = {'A':A.detach().clone(),
           'v':v.detach().clone(),
           'xv':[x.detach().clone() for x in xv]}
    if slice_matching:
        out['A2d'] = A2d
    if full_outputs:
        out['WM'] = WM.detach().clone()
        out['WA'] = WA.detach().clone()
        out['WB'] = WB.detach().clone()
        out['W0'] = W0.detach().clone()
        out['muB'] = muB.detach().clone()
        out['muA'] = muA.detach().clone()
        out['sigmaB'] = sigmaB.detach().clone()
        out['sigmaA'] = sigmaA.detach().clone()
        out['sigmaM'] = sigmaM.detach().clone()
        # others ...
    return out




    
    
    
# this cell needs to get wrapped into a function
# todo this should output step sizes so I can restart with them

# 
# everything in the config will be either a list of the same length as downI
# or a list of length 1
# or a scalar 
def emlddmm_multiscale(**kwargs):
    
    
    # how many levels?
    if 'downI' in kwargs:
        downI = kwargs['downI']
        if type(downI[0]) == list:
            nscales = len(downI)
        else:
            nscales = 1
        
    



    for d in range(nscales):
        # now we have to convert the kwargs to a new dictionary with only one value
        params = {}
        for key in kwargs:
            test = kwargs[key]
                        
            
            # general cases
            if type(test) == list:
                if len(test) > 1:
                    params[key] = test[d]
                else:
                    params[key] = test[0]
            else: # not a list, e.g. inputing v as a numpy array
                params[key] = test            

        output = emlddmm(v_res_factor=3.0,                         
                         sigmaM=np.ones(kwargs['J'].shape[0]),                         
                         sigmaB=np.ones(kwargs['J'].shape[0])*2.0,                         
                         sigmaA=np.ones(kwargs['J'].shape[0])*5.0,                         
                         full_outputs=True,
                        **params)

        A = output['A']
        v = output['v']
        A2d = output['A2d']
        kwargs['A'] = A
        kwargs['v'] = v
        kwargs['A2d'] = A2d
    return output

    
# we need to output the transformations as vtk, with their companion jsons (TODO)
# note Data with implicit topology (structured data such as vtkImageData and vtkStructuredGrid) 
# are ordered with x increasing fastest, then y,thenz .
# this is notation, it means they expect first index fastest in terms of their notation
# I am ignoring the names xyz, and just using fastest to slowest
# note dataTypeis  one  of  the  types
# bit,unsigned_char,char,unsigned_short,short,unsigned_int,int,unsigned_long,long,float,ordouble.
# I should only need the latter 2
dtypes = {    
        np.dtype('float32'):'float',
        np.dtype('float64'):'double',
    }

def write_vtk_data(fname,x,out,title,names=None):
    '''
    Write data as vtk file.
    each channel is saved as a dataset (time for velocity field, or image channel for images)
    each channel is saved as a structured points with a vector or a scalar at each point        
    
    fname is filename to write
    x is voxel locations along last three axes
    if out is size nt x 3 x slices x height x width we assume vector
    if out is size n x slices x height x width we assum scalar     
    title is the name of the dataset
    out needs to be numpy
    names can be a list of names for each dataset or None
    
    note must be BIG endian
    
    only structured points supported, scalars or vectors data type
    '''
    
    if len(out.shape) == 5 and out.shape[1] == 3:
        type_ = 'VECTORS'
    elif len(out.shape) == 4:
        type_ = 'SCALARS'
    else:
        raise Exception('out is not the right size')
    if names is None:        
        names = [f'data_{t:03d}(b)' for t in range(out.shape[0])]

    with open(fname,'wt') as f:
        f.writelines([
            '# vtk DataFile Version 3.0\n',
            title+'\n',
            'BINARY\n',
            'DATASET STRUCTURED_POINTS\n',
            f'DIMENSIONS {out.shape[-1]} {out.shape[-2]} {out.shape[-3]}\n',
            f'ORIGIN {x[-1][0]} {x[-2][0]} {x[-3][0]}\n',
            f'SPACING {x[-1][1]-x[-1][0]} {x[-2][1]-x[-2][0]} {x[-3][1]-x[-3][0]}\n',
            f'POINT_DATA {out.shape[-1]*out.shape[-2]*out.shape[-3]}\n'                  
            ])
    

    for i in range(out.shape[0]):
        with open(fname,'at') as f:
            f.writelines([
                f'{type_} {names[i]} {dtypes[out.dtype]}\n'
            ])
        with open(fname,'ab') as f:
            # make sure big endian 
            if type_ == 'VECTORS':
                out_ = np.array(out[i].transpose(1,2,3,0))
            else:
                out_ = np.array(out[i])
            outtype = np.dtype(out_.dtype).newbyteorder('>')
            out_.astype(outtype).tofile(f)
        with open(fname,'at') as f:
            f.writelines([
                '\n'
            ])
            
def write_matrix_data(fname,A):
    with open(fname,'wt') as f:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):                
                f.write(f'{A[i,j]}, ')
            f.write('\n')

# write outputs
def write_transform_outputs(output_dir, outputs):
    xv = output['xv']
    v = output['v']
    A = output['A']
    A2d = output['A2d']
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir,'transforms/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    output_name = os.path.join(output_dir, 'velocity.vtk')
    title = 'velocity_field'
    write_vtk_data(output_name,xv,v.cpu().numpy(),title)  
    output_name = os.path.join(output_dir, 'A.txt')
    write_matrix_data(output_name,A)
    for i in range(A2d.shape[0]):
        output_name = os.path.join(output_dir,f'A2d_{i:04d}.txt')
        write_matrix_data(output_name,A2d[i])

def write_qc_outputs(output_dir,output,xI,I,xJ,J,xS=None,S=None):
    ''' 
    Write outputs
    '''
    xv = output['xv']
    v = output['v'].detach()
    A = output['A'].detach()
    
    slice_matching = 'A2d' in output
    if slice_matching:
        A2d = output['A2d'].detach()

    Ai = torch.inverse(A)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir,'qc/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # first, lets see the transformed atlas and target
    XJ = torch.stack(torch.meshgrid(xJ))
    A2di = torch.inverse(A2d)
    if slice_matching:
        XJ_ = torch.clone(XJ)    
        XJ_[1:] = ((A2di[:,None,None,:2,:2]@ (XJ[1:].permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)            
    else:
        XJ_ = XJ

    # sample points for affine
    Xs = ((Ai[:3,:3]@XJ_.permute((1,2,3,0))[...,None])[...,0] + Ai[:3,-1]).permute((3,0,1,2))
    # for diffeomorphism
    XV = torch.stack(torch.meshgrid(xv))
    phii = v_to_phii(xv,v)
    phiiAi = interp(xv,phii-XV,Xs) + Xs

    # transform image
    AphiI = interp(xI,I,phiiAi)       


    fig = draw(AphiI,xJ)
    fig[0].suptitle('I input')
    fig[0].savefig(output_dir+'I_input.jpg')

    fig = draw(J,xJ)
    fig[0].suptitle('J input')
    fig[0].savefig(output_dir+'J_input.jpg')

    #fig = draw(torch.cat((AphiI,J,AphiI),0),xJ)
    #fig[0].suptitle('Input Space')
    
    
    # first we need to build the reconstructed space
    # that is, only apply the 2D transforms
    # then I can
    # I'll have to build an intelligent sample space
    # because they may be shifted out of their original volumes
    # to do this we'll apply A2di to the corners of each slice
    # and get the min and max
    XJ = torch.stack(torch.meshgrid(xJ),-1)
    A2di = torch.inverse(A2d)
    points = (A2di[:,None,None,:2,:2]@XJ[...,1:,None])[...,0]
    m0 = torch.min(points[...,0])
    M0 = torch.max(points[...,0])
    m1 = torch.min(points[...,1])
    M1 = torch.max(points[...,1])
    # construct a recon domain
    xr0 = torch.arange(float(m0),float(M0),dJ[1],device=m0.device,dtype=m0.dtype)
    xr1 = torch.arange(float(m1),float(M1),dJ[2],device=m0.device,dtype=m0.dtype)
    xr = xJ[0],xr0,xr1
    XR = torch.stack(torch.meshgrid(xr),-1)
    # now we have to sample J at A Xr
    Xs = torch.clone(XR)
    Xs[...,1:] = (A2d[:,None,None,:2,:2]@XR[...,1:,None])[...,0] + A2d[:,None,None,:2,-1]
    Xs = Xs.permute(3,0,1,2)
    Jr = interp(xJ,J,Xs)
    fig = draw(Jr,xr)
    fig[0].suptitle('J recon')
    fig[0].savefig(output_dir + 'J_recon.jpg')

    # and we need atlas recon
    # sample points for affine
    Xs = ((Ai[:3,:3]@XR[...,None])[...,0] + Ai[:3,-1]).permute((3,0,1,2))
    # for diffeomorphism
    XV = torch.stack(torch.meshgrid(xv))
    phiiAi = interp(xv,phii-XV,Xs) + Xs

    # transform image
    AphiI = interp(xI,I,phiiAi)       
    fig = draw(AphiI,xr)
    fig[0].suptitle('I recon')
    fig[0].savefig(output_dir + 'I_recon.jpg')


    #fig = draw(torch.cat((AphiI,Jr,AphiI),0),xr)
    #fig[0].suptitle('Recon space')
    
    # and atlas space
    XI = torch.stack(torch.meshgrid(xI))
    phi = v_to_phii(xv,-v.flip(0))
    #A = LT_to_A(L.detach(),T.detach())
    Aphi = ((A[:3,:3]@phi.permute((1,2,3,0))[...,None])[...,0] + A[:3,-1]).permute((3,0,1,2))
    Aphi = interp(xv,Aphi,XI)


    phiiAiJ = interp(xr,Jr,Aphi)

    fig = draw(phiiAiJ,xI)
    fig[0].suptitle('J atlas')
    fig[0].savefig(output_dir+'J_atlas.jpg' )

    fig = draw(I,xI)
    fig[0].suptitle('I atlas')
    fig[0].savefig(output_dir+'I_atlas.jpg')

    #fig = draw(torch.cat((I,phiiAiJ,I),0),xI)
    #fig[0].suptitle('atlas space')

    output_slices = slice_matching and ( (xS is not None) and (S is not None))
    if output_slices:
        # transform S
        AphiS = interp(xS,torch.tensor(S[None].astype(float),device=device,dtype=dtype),phiiAi,mode='nearest').cpu().numpy()[0]

        mods = (7,11,13)
        R = (AphiS%mods[0])/mods[0]
        G = (AphiS%mods[1])/mods[1]
        B = (AphiS%mods[2])/mods[2]
        #fig = draw(np.stack((R,G,B)),xr)

        # also outlines
        M = np.zeros_like(AphiS)
        r = 1
        for i in [0]:#range(-r,r+1): # in the coronal plane my M is "nice"
            for j in range(-r,r+1):
                for k in range(-r,r+1):
                    if (i**2 + j**2 + k**2) > r**2:
                        continue
                    M = np.logical_or(M,np.roll(AphiS,shift=(i,j,k),axis=(0,1,2))!=AphiS)
        #fig = draw(M[None])

        C = np.stack((R,G,B))*M
        q = (0.01,0.99)
        c = np.quantile(J.cpu().numpy(),q)
        Jn = (Jr.cpu().numpy() - c[0])/(c[1]-c[0])
        Jn[Jn<0] = 0
        Jn[Jn>1] = 1
        alpha = 0.5
        show_ = Jn[0][None]*(1-M[None]*alpha) + M[None]*C*alpha


        #fig = plt.figure(figsize=(8,10))
        #draw(show_,n_slices=3,fig=fig)

        #f,ax = plt.subplots()
        #ax.imshow(show_[:,show.shape[1]//2].transpose(1,2,0))
        
        f,ax = plt.subplots()
        for s in range(show_.shape[1]):
            ax.cla()
            ax.imshow(show_[:,s].transpose(1,2,0))
            ax.set_xticks([])
            ax.set_yticks([])
            f.savefig(os.path.join(output_dir,f'slice_{s:04d}.jpg'))


