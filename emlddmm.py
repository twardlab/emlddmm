import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample
import glob
import os
import nibabel
import nrrd
import json
import re
import argparse
import warnings



# display
def draw(J,xJ=None,fig=None,n_slices=5,vmin=None,vmax=None,**kwargs):    
    """ Draw 3D imaging data.
    
    Images are shown by sampling slices along 3 orthogonal axes.
    Color or grayscale data can be shown.
    
    Parameters
    ----------
    J : array like (torch tensor or numpy array)
        A 3D image with C channels should be size (C x nslice x nrow x ncol)
        Note grayscale images should have C=1, but still be a 4D array.
    xJ : list
        A list of 3 numpy arrays.  xJ[i] contains the positions of voxels
        along axis i.  Note these are assumed to be uniformly spaced. The default
        is voxels of size 1.0.
    fig : matplotlib figure
        A figure in which to draw pictures. Contents of hte figure will be cleared.
        Default is None, which creates a new figure.
    n_slices : int
        An integer denoting how many slices to draw along each axis. Default 5.
    vmin
        A minimum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.001 quantile on each channel.
    vmax
        A maximum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.999 quantile on each channel.
    kwargs : dict
        Other keywords will be passed on to the matplotlib imshow function. For example
        include cmap='gray' for a gray colormap

    Returns
    -------
    fig : matplotlib figure
        The matplotlib figure variable with data.
    axs : array of matplotlib axes
        An array of matplotlib subplut axes containing each image.


    """
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
        ax.imshow(J[:,slices[i]].transpose(1,2,0).squeeze(),vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax1
    slices = np.round(np.linspace(0,J.shape[2],n_slices+2)[1:-1]).astype(int)    
    extent = (xJ[2][0],xJ[2][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1+n_slices)
        ax.imshow(J[:,:,slices[i]].transpose(1,2,0).squeeze(),vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax2
    slices = np.round(np.linspace(0,J.shape[3],n_slices+2)[1:-1]).astype(int)    
    extent = (xJ[1][0],xJ[1][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):        
        ax = fig.add_subplot(3,n_slices,i+1+n_slices*2)
        ax.imshow(J[:,:,:,slices[i]].transpose(1,2,0).squeeze(),vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    
    return fig,axs
    
    
def load_slices(target_name):
    """ Load a slice dataset.
    
    Load a slice dataset for histology registration. Slice datasets include pairs
    of images and json sidecar files, as well as one tsv file explaining the dataset.
    Note this code creates a 3D array by padding.
    
    Parameters
    ----------
    target_name : string
        Name of a directory containing slice dataset.
        
    Returns
    -------
    xJ : list of numpy arrays
        Location of v
    J : numpy array
        Numpy array of size C x nslices x nrows x ncols where C is the number of channels
        e.g. C=3 for RGB.
    W0 : numpy array
        A nslices x nrows x ncols numpy array containing weights.  Weights are 0 where there 
        was padding
    
    References
    ----------
    Document describing dataset format here: TODO XXXXX
    documented XXXX
    
    """
    print('loading target images')
    fig,ax = plt.subplots()
    ax = [ax]
    # current limitation
    # requires the word 'present'
    # requires the first image to be present
    # expects data type to be in 0,1
    # assumes space directions are diagonal
    # todo: origin
    # we will need more control over the size, and we will need to maintain the origin of each slice
    # right now we have a heuristic for taking 99th percentile and expanding by 1%
    
    data = []
    # load the one tsv file
    tsv_name = os.path.join(target_name, 'samples.tsv' )
    with open(tsv_name,'rt') as f:
        for count,line in enumerate(f):
            line = line.strip()
            key = '\t' if '\t' in line else '    '
            if count == 0:
                headings = re.split(key,line)                
                continue
            data.append(re.split(key,line))
    data_ = np.zeros((len(data),len(data[0])),dtype=object)
    for i in range(data_.shape[0]):
        for j in range(data_.shape[1]):
            try:
                data_[i,j] = data[i][j]
            except:
                data_[i,j] = ''
    data = data_
    
    # now we will loop through the files and get the sizes 
    nJ_ = np.zeros((data.shape[0],3),dtype=int)
    J_ = []
    for i in range(data.shape[0]):
        namekey = data[i,0]        
        searchstring = os.path.join(target_name,'*'+os.path.splitext(namekey)[0]+'*.json')        
        jsonfile = glob.glob(searchstring)
        present = data[i,3] == 'present'
        if not present:
            if i == 0:
                raise Exception('First image is not present')
            J_.append(np.array([[[0.0,0.0,0.0]]]))
            continue
        with open(jsonfile[0]) as f:
            jsondata = json.load(f)
        #nJ_[i] = np.array(jsondata['Sizes'])
        
        
        # this should contain an image and a json    
        if 'DataFile' in jsondata:
            image_name = jsondata['DataFile']
        elif 'DataFIle' in jsondata:
            image_name = jsondata['DataFIle']
        try:
            J__ = plt.imread(os.path.join(target_name,image_name))
        except:
            J__ = plt.imread(image_name)
        if J__.dtype == np.uint8:
            J__ = J__.astype(float)/255.0
        
        J__ = J__[...,:3] # no alpha        
        nJ_[i] = np.array(J__.shape)
        
        J_.append(J__)
        
        if not i%20:
            ax[0].cla()
            ax[0].imshow(J__)

                  
            fig.suptitle(f'slice {i} of {data.shape[0]}: {image_name}')
            fig.canvas.draw()
                
        # the domain
        if i == 0:
            dJ = np.diag(np.array(jsondata['SpaceDirections'][1:]))[::-1]
            
        # note the order needs to be reversed
        
        # if this is the first file we want to set up a 3D volume
        
    nJm = np.max(nJ_,0)
    nJm = (np.quantile(nJ_,0.95,axis=0)*1.01).astype(int) 
    # this will look for outliers when there are a small number, 
    # really there just shouldn't be outliers
    nJsave = np.copy(nJ_) 
    
    print('padding and assembling into 3D volume')
    J = np.zeros((len(data),nJm[0],nJm[1],3))
    W0 = np.zeros((len(data),nJm[0],nJm[1]))
    for i in range(len(J_)):
        J__ = J_[i]
        topad = nJm - np.array(nJ_[i])
        # just pad on the left and I'll fill it in
        topad = np.array(((topad[0]//2,0),(topad[1]//2,0),(0,0)))
        # if there are any negative values I need to crop, I'll just crop on the right
        if np.any(np.array(topad)<0):
            if topad[0][0] < 0:
                J__ = J__[:J.shape[1]]
                topad[0][0] = 0
            if topad[1][0] < 0:
                J__ = J__[:,:J.shape[2]]
                topad[1][0] = 0
        Jp = np.pad(J__,topad,constant_values=np.nan)
        W0_ = np.logical_not(np.isnan(Jp[...,0]))
        Jp[np.isnan(Jp)] = 0
        W0[i,:W0_.shape[0],:W0_.shape[1]] = W0_
        J[i,:W0_.shape[0],:W0_.shape[1],:] = Jp
    J = np.transpose(J,(3,0,1,2))    
    nJ = np.array(J.shape)
    xJ = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(nJ[1:],dJ)]
    W0 = W0 * np.logical_not(np.all(J==0.0,0))
    W0 = W0 * np.logical_not(np.all(J==1.0,0))
    # free up memory, not sure if this is necessary inside a function
    del J_
    return xJ,J,W0
    

            
# resampling
def sinc_resample_numpy(I,n):
    ''' Perform sinc resampling of an image in numpy.
    
    This function does sinc resampling using numpy rfft
    torch does not let us control behavior of fft well enough
    This is intended to be used to resample velocity fields if necessary
    Only intending it to be used for upsampling.
    
    Parameters
    ----------
    I : numpy array
        An image to be resampled. Can be an arbitrary number of dimensions.
    n : list of ints
        Desired dimension of output data.
    
    
    Returns
    -------
    Id : numpy array
        A resampled image of size n.
    '''
    Id = np.array(I)
    for i in range(len(n)):        
        if I.shape[i] == n[i]:
            continue
        Id = np.fft.irfft(np.fft.rfft(Id,axis=i),axis=i,n=n[i])
    # output with correct normalization
    Id = Id*np.prod(Id.shape)/np.prod(I.shape) 
    return Id
    

def downsample_ax(I,down,ax):
    '''
    Downsample imaging data along one of the first 5 axes.
    
    Imaging data is downsampled by averaging nearest pixels.
    Note that data will be lost from the end of images instead of padding.
    This function is generally called repeatedly on each axis.
    
    Parameters
    ----------
    I : array like (numpy or torch)
        Image to be downsampled on one axis.
    down : int
        Downsampling factor.  2 means average pairs of nearest pixels 
        into one new downsampled pixel
    ax : int
        Which axis to downsample along.
    
    Returns
    -------
    Id : array like
        The downsampled image.
    '''
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
    Id = Id/down
    return Id
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
    '''
    Downsample an image as well as pixel locations
    
    Parameters
    ----------
    xI : list of numpy arrays
        xI[i] is a numpy array storing the locations of each voxel
        along the i-th axis.
    I : array like
        Image to be downsampled
    down : list of ints
        Factor by which to downsample along each dimension
        
    Returns
    -------
    xId : list of numpy arrays
        New voxel locations in the same format as xI
    Id : numpy array
        Downsampled image.
    '''
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
    Interpolate a 3D image with specified regular voxel locations at specified sample points.
    
    Interpolate the 3D image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D arrays with first channel storing component)
    
    Parameters
    ----------
    x : list of numpy arrays
        x[i] is a numpy array storing the pixel locations of imaging data along the i-th axis.
        Note that this MUST be regularly spaced, only the first and last values are queried.
    I : array
        Numpy array or torch tensor storing 3D imaging data.  I is a 4D array with 
        channels along the first axis and spatial dimensions along the last 3 
    phii : array
        Numpy array or torch tensor storing positions of the sample points. phii is a 4D array
        with components along the first axis (e.g. x0,x1,x1) and spatial dimensions 
        along the last 3.
    kwargs : dict
        keword arguments to be passed to the grid sample function. For example
        to specify interpolation type like nearest.  See pytorch grid_sample documentation.
    
    Returns
    -------
    out : torch tensor
        4D array storing a 3D image with channels stored along the first axis. 
        This is the input image resampled at the points stored in phii.
    
    
    '''
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
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
    out = out[0]
    return out
    
# now we need to create a flow
# timesteps will be along the first axis
def v_to_phii(xv,v):
    '''
    Use Euler's method to construct a position field from a velocity field
    by integrating over time.
    
    This method uses interpolation and subtracts and adds identity for better
    behavior outside boundaries. This method is sometimes refered to as the
    method of characteristics.
    
    Parameters
    ----------
    xv : list of 1D tensors
        xv[i] is a tensor storing the location of the sample points along
        the i-th dimension of v
    v : 5D tensor
        5D tensor where first axis corresponds to time, second corresponds to 
        component, and 3rd to 5th correspond to spatial dimensions.
    
    Returns
    -------    
    phii : 4D tensor
        Inverse transformation is output with component on the first dimension
        and space on the last 3. Note that the whole timeseries is not output.
    
    '''
    XV = torch.stack(torch.meshgrid(xv))
    phii = torch.clone(XV)
    dt = 1.0/v.shape[0]
    for t in range(v.shape[0]):
        Xs = XV - v[t]*dt
        phii = interp(xv,phii-XV,Xs)+Xs
    return phii
        
def emlddmm(**kwargs):
    '''
    Run the EMLDDMM algorithm for deformable registration between two
    different imaging modalities with possible missing data in one of them
    
    Details of this algorithm can be found in
    [1] Tward, Daniel, et al. "Diffeomorphic registration with intensity 
    transformation and missing data: Application to 3D digital pathology 
    of Alzheimer's disease." Frontiers in neuroscience 14 (2020): 52.
    [2] Tward, Daniel, et al. "3d mapping of serial histology sections 
    with anomalies using a novel robust deformable registration algorithm." 
    Multimodal Brain Image Analysis and Mathematical Foundations of 
    Computational Anatomy. Springer, Cham, 2019. 162-173.
    [3] Tward, Daniel, et al. "Solving the where problem in neuroanatomy: 
    a generative framework with learned mappings to register multimodal, 
    incomplete data into a reference brain." bioRxiv (2020).
    
    Parameters
    ----------
    Note all parameters are keyword arguments, but the first four are required.    
    xI : list of arrays
        xI[i] stores the location of voxels on the i-th axis of the atlas image I (REQUIRED)
    I : 4D array (numpy or torch)
        4D array storing atlas imaging data.  Channels (e.g. RGB are stored on the 
        first axis, and the last three are spatial dimensions. (REQUIRED)
    xJ : list of arrays
        xJ[i] stores the location of voxels on the i-th axis of the target image J (REQUIRED)
    J : 4D array (numpy or torch)
        4D array storing target imaging data.  Channels (e.g. RGB are stored on the 
        first axis, and the last three are spatial dimensions. (REQUIRED)
    nt : int
        Number of timesteps for integrating a velocity field to yeild a position field (default 5).
    eA : float
        Gradient descent step size for affine component (default 1e-5).  It is strongly suggested
        that you test this value and not rely on defaults. Note linear and translation components
        are combined following [ref] so only one stepsize is required.
    ev : float
        Gradient descent step size for affine component (default 1e-5).  It is strongly suggested
        that you test this value and not rely on defaults.
    
        
    
    
    '''
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
            g[i,j] = torch.sum(EiX*EjX) * torch.prod(dI) # because gradient has a factor of N in it, I think its a good idea to do sum
    # note, on july 21 I add factor of voxel size, so it can cancel with factor in cost function
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
            g2d[i,j] = torch.sum(EiX*EjX) * torch.prod(dI[1:])
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
            A2d = torch.as_tensor(A2d, device=device, dtype=dtype)
            A2d.requires_grad = True
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
            # so far it looks like it is exactly the same as the above
            # I need to update
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
        version_num = int(torch.__version__[2])
        if version_num < 7:
            vhat = torch.rfft(v,3,onesided=False)
        else:
            vhat = torch.view_as_real(torch.fft.fftn(v,dim=3,norm="backward"))
        
        ER = torch.sum(torch.sum(vhat**2,(0,1,-1))*LL)/torch.prod(nv)*torch.prod(dv)/nt/2.0/sigmaR**2
        

        # total cost 
        E = EM + ER
        

        # gradient
        E.backward()

        # covector to vector
        if version_num < 7:
            vgrad = torch.irfft(torch.rfft(v.grad,3,onesided=False)*(KK)[None,None,...,None],3,onesided=False)
        else:
            vgrad = torch.view_as_real(torch.fft.ifftn(torch.fft.fftn(v.grad,dim=3,norm="backward")*(KK),
                dim=3,norm="backward"))
            vgrad = vgrad[...,0]
        
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


        if not it%n_draw or it==n_iter-1:
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
        # return figures
        out['figA'] = figA
        #out['figA'].savefig(os.path.join(output_dir,'A'))
        out['figE'] = figE
        #out['figE'].savefig(os.path.join(output_dir,'costminimization'))        
        out['figI'] = figI
        #out['figI'].savefig(os.path.join(output_dir,'I'))        
        out['figfI'] = figfI
        #out['figfI'].savefig(os.path.join(output_dir,'fI'))        
        out['figErr'] = figErr
        #out['figErr'].savefig(os.path.join(output_dir,'errors'))        
        out['figJ'] = figJ
        #out['figJ'].savefig(os.path.join(output_dir,'J'))
        out['figW'] = figW
        #out['figW'].savefig(os.path.join(output_dir,'W'))
        out['figV'] = figV
        #out['figV'].savefig(os.path.join(output_dir,'V'))
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
    # note I expect downI to be either a list, or a list of lists, not numpy array
    if 'downI' in kwargs:
        downI = kwargs['downI']        
        if type(downI[0]) == list:
            nscales = len(downI)
        else:
            nscales = 1
        print(f'Found {nscales} scales')
        
        
    
    outputs = []
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
        
        if 'sigmaM' not in params:
            params['sigmaM'] = np.ones(kwargs['J'].shape[0])
        if 'sigmaB' not in params:
            params['sigmaB'] = np.ones(kwargs['J'].shape[0])*2.0
        if 'sigmaA' not in params:
            params['sigmaA'] = np.ones(kwargs['J'].shape[0])*5.0
        output = emlddmm(v_res_factor=3.0,      
                         full_outputs=True,
                        **params)
        # I should save an output at each iteration
        outputs.append(output)

        A = output['A']
        v = output['v']
        
        kwargs['A'] = A
        kwargs['v'] = v
        if 'slice_matching' in params and params['slice_matching']:
            A2d = output['A2d']
            kwargs['A2d'] = A2d
        
    return outputs # should I return the whole list outputs?

    
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
        np.dtype('uint8'):'unsigned_char',
        np.dtype('uint16'):'unsigned_short',
        np.dtype('uint32'):'unsigned_int',
        np.dtype('uint64'):'unsigned_long',
        np.dtype('int8'):'char',
        np.dtype('int16'):'short',
        np.dtype('int32'):'int',
        np.dtype('int64'):'long',
    }


def write_vtk_data(fname,x,out,title,names=None):
    
    '''
    Write data as vtk file.
    inputs should be numpy, but will check for tensor
    each channel is saved as a dataset (time for velocity field, or image channel for images)
    each channel is saved as a structured points with a vector or a scalar at each point        
    
    fname is filename to write
    x is voxel locations along last three axes
    if out is size nt x 3 x slices x height x width we assume vector
    if out is size n x slices x height x width we assume scalar     
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
    else:
        # make sure we know it is big endian
        names = [n if '(b)' in n else n+'(b)' for n in names]
        
        

    if type(out) == torch.Tensor:        
        out = out.cpu().numpy()
    
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
                f.write('LOOKUP_TABLE default\n'.encode())
                out_ = np.array(out[i])
            outtype = np.dtype(out_.dtype).newbyteorder('>')
            out_.astype(outtype).tofile(f)
        with open(fname,'at') as f:
            f.writelines([
                '\n'
            ])
            
dtypes_reverse = {
    'float':np.dtype('float32'),
    'double':np.dtype('float64'),
    'unsigned_char':np.dtype('uint8'),
    'unsigned_short':np.dtype('uint16'),
    'unsigned_int':np.dtype('uint32'),
    'unsigned_long':np.dtype('uint64'),
    'char':np.dtype('int8'),
    'short':np.dtype('int16'),
    'int':np.dtype('int32'),
    'long':np.dtype('int64'),
}    
def read_vtk_data(fname,endian='b'):
    '''
    Read data back in same format, I will check that I get identity
    
    Note endian should always be big, but we support little as well
    
    input is filename and optionally endian
    
    output is x,images,title,names
    
    only legacy format supported
    '''
    # TODO support skipping blank lines
    big = not (endian=='l')
    
    verbose = True
    verbose = False
    with open(fname,'rb') as f:        
        # first line should say vtk version
        line = f.readline().decode().strip()
        if verbose: print(line)
        if 'vtk datafile' not in line.lower():
            raise Exception('first line should include vtk DataFile Version X.X')
        # second line says title    
        line = f.readline().decode().strip()
        if verbose: print(line)
        title = line

        # third line should say type of data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if not line.upper() == 'BINARY':
            raise Exception(f'Only BINARY data type supported, but this file contains {line}')
        data_format = line

        # next line says type of data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if not line.upper() == 'DATASET STRUCTURED_POINTS':
            raise Exception(f'Only STRUCTURED_POINTS dataset supported, but this file contains {line}')
        geometry = line

        # next line says dimensions    
        # "ordered with x increasing fastest, theny,thenz"
        # this is the same as nrrd (fastest to slowest)
        # however our convention in python that we use channel z y x order
        # i.e. the first is channel
        line = f.readline().decode().strip()
        if verbose: print(line)
        dimensions = np.array([int(n) for n in line.split()[1:]])
        if len(dimensions) not in [3,4]:
            raise Exception(f'Only datasets with 3 or 4 axes supported, but this file contains {dimensions}')

        # next says origin
        line = f.readline().decode().strip()
        if verbose: print(line)
        origin = np.array([float(n) for n in line.split()[1:]])

        # next says spacing
        line = f.readline().decode().strip()
        if verbose: print(line)
        spacing = np.array([float(n) for n in line.split()[1:]])

        # now I can build axes
        # note I have to reverse the order for python
        x = [np.arange(n)*d+o for n,d,o in zip(dimensions[::-1],spacing[::-1],origin[::-1])]

        # next line must contain point_data
        line = f.readline().decode().strip()
        if verbose: print(line)
        if 'POINT_DATA' not in line:
            raise Exception(f'only POINT_DATA supported but this file contains {line}')                          
        N = int(line.split()[-1])

        # now we will loop over available datasets
        names = []
        images = []
        count = 0
        while True:
            
            # first line contains data type (scalar or vector), name, and format
            # it could be a blank line
            line = f.readline().decode()
            if line == '\n':
                line = f.readline().decode()        
            line = line.strip()
            
            if line is None or not line: # check if we've reached the end of the file
                break
                
            if verbose: print(f'starting to load dataset {count}')
                
            if verbose: print(line)            
            S_V = line.split()[0]
            name = line.split()[1]
            dtype = line.split()[2]
            names.append(name)

            if S_V.upper() not in ['SCALARS','VECTORS']:
                raise Exception(f'Only scalars or vectors supported but this file contains {S_V}')        
            
            if '(b)' not in name and big: 
                warnings.warn(f'Note (b) symbol not in data name {name}, you should check that it was written big endian. Specify endian="l" if you want little')
                            
            dtype_numpy = dtypes_reverse[dtype]
            if big:
                dtype_numpy_big = dtype_numpy.newbyteorder('>') # > means big endian
            else:
                dtype_numpy_big = dtype_numpy
            #
            # read the data
            if S_V == 'SCALARS':
                # there should be a line with lookup table
                line = f.readline().decode()
                if verbose: print(line)
                data = np.fromfile(f,dtype_numpy_big,N).astype(dtype_numpy)
                # shape it
                data = data.reshape(dimensions[::-1])
                # axis order is already correct because of slowest to fastest convention in numpy

            elif S_V == 'VECTORS':            
                data = np.fromfile(f,dtype_numpy_big,N*3).astype(dtype_numpy)
                # shape it
                data = data.reshape((dimensions[-1],dimensions[-2],dimensions[-3],3))
                # move vector components first
                data = data.transpose((3,0,1,2))
            images.append(data)
            count += 1
        images = np.stack(images) # stack on axis 0
    return x,images,title,names
    
    
def read_data(fname,**kwargs):    
    '''
    Read array data from several file types.
    
    This function will read array based data of several types
    and output x,images,title,names. Note we prefer vtk legacy format, 
    but accept some other formats as read by nibabel.
    
    Parameters
    ----------
    fname : str
        Filename (full path or relative) of array data to load. Can be .vtk or 
        nibabel supported formats (e.g. .nii)
        
    **kwargs : dict
        Keword parameters that are passed on to the loader function
    
    Returns
    -------
    
    x : list of numpy arrays
        Pixel locations where each element of the list identifies pixel
        locations in corresponding axis.
    images : numpy array
        Imaging data of size channels x slices x rows x cols, or of size
        time x 3 x slices x rows x cols for velocity fields
    title : str
        Title of the dataset (read from vtk files)        
    names : list of str
        Names of each dataset (channel or time point)
    
    '''
    # find the extension
    # if no extension use slice reader
    # if vtk use our reader
    # if nrrd use nrrd
    # otherwise try nibabel
    base,ext = os.path.splitext(fname)
    if ext == '.gz':
        base,ext_ = os.path.splitext(base)
        ext = ext_+ext
    print(f'Found extension {ext}')
    
    if ext == '':
        xJ,J,W0 = load_slices(fname)
        x = xJ
        images = np.concatenate((J,W0[None]))
        # set the names, I will separate out mask later
        names = ['red','green','blue','mask']
        title = 'slice_dataset'
    elif ext == '.vtk':
        x,images,title,names = read_vtk_data(fname,**kwargs)
    elif ext == '.nrrd':
        print('opening with nrrd')
        raise Exception('NRRD not currently supported')
    else:
        print('Opening with nibabel, note only 3D images supported')
        vol = nibabel.load(fname,**kwargs)
        images = np.array(vol.get_fdata())
        if images.ndim == 3:
            images = images[None]
        A = vol.header.get_base_affine()
        if not np.allclose(np.diag(np.diag(A[:3,:3])),A[:3,:3]):
            raise Exception('Only support diagonal affine matrix with nibabel')
        x = [ A[i,-1] + np.arange(images.shape[i+1])*A[i,i] for i in range(3)]
        for i in range(3):
            if A[i,i] < 0:
                x[i] = x[i][::-1]
                images = np.array(np.flip(images,axis=i+1))
            
        title = ''
        names = ['']
        
    return x,images,title,names
def write_data(fname,x,out,title,names=None):
    base,ext = os.path.splitext(fname)
    if ext == '.gz':
        base,ext_ = os.path.splitext(base)
        ext = ext_+ext
    print(f'Found extension {ext}')
    
    if ext == '.vtk':
        write_vtk_data(fname,x,out,title,names)
    elif ext == '.nii' or ext == '.nii.gz':
        if type(out) == torch.Tensor:
            out = out.cpu().numpy()
        if out.ndim == 4 and out.shape[0]==1:
            out = out[0]
        if out.ndim >= 4:
            raise Exception('Only grayscale images supported in nii format')
            
        affine = np.diag((x[0][1]-x[0][0],x[1][1]-x[1][0],x[2][1]-x[2][0],1.0))
        affine[:3,-1] = np.array((x[0][0],x[1][0],x[2][0]))
        img = nibabel.Nifti1Image(out, affine)
        nibabel.save(img, fname)  
        warnings.warn('Writing image in nii fomat, no title or names saved')

        
    else:
        raise Exception('Only vtk and .nii/.nii.gz outputs supported')
        
    

def write_matrix_data(fname,A):
    with open(fname,'wt') as f:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):                
                f.write(f'{A[i,j]}, ')
            f.write('\n')

# write outputs
def write_transform_outputs(output_dir, output):
    xv = output['xv']
    v = output['v']
    A = output['A']
    slice_outputs = 'A2d' in output
    if slice_outputs:
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
    if slice_outputs:
        for i in range(A2d.shape[0]):
            output_name = os.path.join(output_dir,f'A2d_{i:04d}.txt')
            write_matrix_data(output_name,A2d[i])

def write_qc_outputs(output_dir,output,xI,I,xJ,J,xS=None,S=None):
    ''' 
    Write outputs
    TODO figure out how to do this with or without A2d
    I still want to output per slice
    '''
    
    xv = [x.to('cpu') for x in output['xv']]
    v = output['v'].detach().to('cpu')
    A = output['A'].detach().to('cpu')
    print(A.device)
    
    
    # 
    device = A.device
    dtype = A.dtype
    
    # to torch
    J = torch.as_tensor(J,dtype=dtype,device=device)
    xJ = [torch.as_tensor(x,dtype=dtype,device=device) for x in xJ]
    I = torch.as_tensor(I,dtype=dtype,device=device)
    xI = [torch.as_tensor(x,dtype=dtype,device=device) for x in xI]
    if S is not None: # segmentations go with atlas, they are integers
        S = torch.as_tensor(S,device=device,dtype=dtype) 
        # don't specify dtype here, you had better set it in numpy
        # actually I need it as float in order to apply interp
        if xS is not None:
            xS = [torch.as_tensor(x,dtype=dtype,device=device) for x in xI]
            
    slice_matching = 'A2d' in output
    if slice_matching:
        A2d = output['A2d'].detach().to('cpu')

    Ai = torch.inverse(A)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir,'qc/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print(f'output dir is {output_dir}')
    
    # first, lets see the transformed atlas and target    
    XJ = torch.stack(torch.meshgrid(xJ))    
    if slice_matching:
        A2di = torch.inverse(A2d)
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
    # this is only done if there is 2D
    # maybe I should have reorientation at least otherwise? Not yet
    # that is, only apply the 2D transforms
    # then I can
    # I'll have to build an intelligent sample space
    # because they may be shifted out of their original volumes
    # to do this we'll apply A2di to the corners of each slice
    # and get the min and max
    XJ = torch.stack(torch.meshgrid(xJ),-1)
    if slice_matching:
        A2di = torch.inverse(A2d)
        points = (A2di[:,None,None,:2,:2]@XJ[...,1:,None])[...,0]
        m0 = torch.min(points[...,0])
        M0 = torch.max(points[...,0])
        m1 = torch.min(points[...,1])
        M1 = torch.max(points[...,1])
        # construct a recon domain
        dJ = [x[1]-x[0] for x in xJ]
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
    else:
        Jr = J
        xr = xJ


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
        # note here I had previously converted it to float
        AphiS = interp(xS,torch.tensor(S,device=device,dtype=dtype),phiiAi,mode='nearest').cpu().numpy()[0]

        mods = (7,11,13)
        R = (AphiS%mods[0])/mods[0]
        G = (AphiS%mods[1])/mods[1]
        B = (AphiS%mods[2])/mods[2]
        fig = draw(np.stack((R,G,B)),xr)

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


class Transform():    
    '''
    A simple class for storing and applying transforms
    TODO: add another type for series of 2D transforms
    '''
    def __init__(self,data,direction='f',domain=None,dtype=torch.float,device='cpu'):
        if type(data) == str:
            # instead we load it
            prefix,extension = os.path.splitext(data)
            if extension == '.txt':
                data = np.genfromtxt(data,delimiter=',')                
                # note that there are nans at the end if I have commas at the end
                if np.isnan(data[0,-1]):
                    data = data[:,:data.shape[1]-1]
                    #print(data)
            elif extension == '.vtk':
                x,images,title,names = read_vtk_data(data)
                domain = x
                data = images
            else:
                raise Exception(f'Only txt and vtk files supported but your transform is {data}')
            
        self.data = torch.as_tensor(data,dtype=dtype,device=device)
        if domain is not None:
            domain = [torch.as_tensor(d,dtype=dtype,device=device) for d in domain]            
        self.domain = domain
        if not direction in ['f','b']:
            raise Exception(f'Direction must be \'f\' or \'b\' but it was \'{direction}\'')
        self.direction = direction
        
        # if it is a velocity field we need to integrate it
        if self.data.ndim == 5:
            if self.domain is None:
                raise Exception('Domain is required when inputting veloctiy field')
            if self.direction == 'b':
                self.data = v_to_phii(self.domain,self.data)
            else:
                self.data = v_to_phii(self.domain,-torch.flip(self.data,(0,)))
        elif self.data.ndim == 2:# if it is a matrix check size
            if self.data.shape[0] == 3 and self.data.shape[1]==3:
                tmp = torch.eye(4,device=device,dtype=dtype)
                tmp[1:,1:] = self.data
                self.data = tmp            
            elif not (self.data.shape[0] == 4 and self.data.shape[1]==4):
                raise Exception(f'Only 3x3 or 4x4 matrices supported now but this is {self.data.shape}')
                
            if self.direction == 'b':
                self.data = torch.inverse(self.data)
        elif self.data.ndim == 4: # if it is a mapping
            if self.direction == 'b':
                raise Exception(f'When specifying a mapping, backwards is not supported')
        
                
                
                
    def apply(self,X):
        X = torch.as_tensor(X,dtype=self.data.dtype,device=self.data.device)
        if self.data.ndim == 2:
            # then it is a matrix
            A = self.data
            return ((A[:3,:3]@X.permute(1,2,3,0)[...,None])[...,0] + A[:3,-1]).permute(3,0,1,2)
        elif self.data.ndim == 4:
            # then it is a mapping, we need interp
            # recall all components are stored on the first axis,
            # but for sampling they need to be on the last axis
            ID = torch.stack(torch.meshgrid(self.domain))
            #print(f'ID shape {ID.shape}')
            #print(f'X shape {X.shape}')
            #print(f'data shape {self.data.shape}')
            return interp(self.domain,(self.data-ID),X) + X
            
        
            
    def __repr__(self):
        return f'Transform with data size {self.data.shape}, direction {self.direction}, and domain {type(self.domain)}'
    def __call__(self,X):
        return self.apply(X)
            
# now wrap this into a function
def compose_sequence(transforms,Xin,direction='f'):
    '''
    Input can be a list of transforms class.
    Or a list of filenames (single direction in argument)
    Or a list of a list of 2 tuples that specify direction (f,b)
    Or an output directory
    
    Note f is default which maps points from atlas to target, 
    or images from target to atlas.
    
    Xin are the points we want to transform (e.g. sample points in atlas)
    
    TODO use os path join
    TODO support direction as a list, right now direction only is used for a single direction
    
    Note, if the input is a string, we assume it is an output directory and get A and V. In this case we use the direction argument.
    If the input is a tuple of length 2, we assume it is an output directory and a direction
    
    Otherwise, the input must be a list.  It can be a list of strings, or transforms, or string-direction tuples.
    
    What if it is a list of length 1?
    
    '''
    
    print(f'starting to compose sequence with transforms {transforms}')    
    
    
    # check special case for a list of length 1 but the input is a directory
    if (type(transforms) == list and len(transforms)==1 
        and type(transforms[0]) == str and os.path.splitext(transforms[0])[1]==''):
        transforms = transforms[0]
    # check special case for a list of length one but the input is a tuple
    if (type(transforms) == list and len(transforms)==1 
        and type(transforms[0]) == tuple and type(transforms[0][0]) == str 
        and os.path.splitext(transforms[0][0])[1]=='' and transforms[0][1].lower() in ['b','f']):
        direction = transforms[0][1]
        transforms = transforms[0][0] # note variable is redefined                
    
    
    if type(transforms) == str or ( type(transforms) == list and length(transforms)==1 and type(transforms[0]) == str ):
        if type(transforms) == list: transforms = transforms[0]            
        # assume output directory
        # print('printing transforms input')
        # print(transforms)
        if direction == 'b':
            # backward, first affine then deform
            transforms = [Transform(os.path.join(transforms,'transforms','A.txt'),direction=direction),
                          Transform(os.path.join(transforms,'transforms','velocity.vtk'),direction=direction)]
        elif direction == 'f':
            # forward, first deform then affine
            transforms = [Transform(os.path.join(transforms,'transforms','velocity.vtk'),direction=direction),
                          Transform(os.path.join(transforms,'transforms','A.txt'),direction=direction)]    
        #print('printing modified transforms')
        #print(transforms)    
    elif type(transforms) == list:
        if type(transforms[0]) == Transform:
            # don't do anything here
            pass
        elif type(transforms[0]) == str:
            # list of strings
            transforms = [Transform(t,direction=direction) for t in transforms]
        elif type(transforms[0]) == tuple:
            transforms = [Transform(t[0],direction=t[1]) for t in transforms]
        else:
            raise Exception('Transforms must be either output directory, \
        or list of objects, or list of filenames, \
        or list of tuples storing filename/direction')
    else:
        raise Exception('Transforms must be either output directory, \
        or list of objects, or list of filenames, \
        or list of tuples storing filename/direction')
    Xin = torch.as_tensor(Xin,device=transforms[0].data.device,dtype=transforms[0].data.dtype)    
    Xout = torch.clone(Xin)
    for t in transforms:
        Xout = t(Xout)
    return Xout
    
def apply_transform_float(x,I,Xout):
    '''Apply transform to image
    Image points stored in x, data stored in I
    transform stored in Xout
    
    There is an issue with numpy integer arrays, I'll have two functions
    '''
    if type(I) == np.array:
        isnumpy = True
    else:
        isnumpy = False
    
    AphiI = interp(x,torch.as_tensor(I,dtype=Xout.dtype,device=Xout.device),Xout).cpu()
    if isnumpy:
        AphiI = IphiI.numpy()
    return AphiI
def apply_transform_int(x,I,Xout):
    '''Apply transform to image
    Image points stored in x, data stored in I
    transform stored in Xout
    
    There is an issue with numpy integer arrays, I'll have two functions
    '''
    if type(I) == np.array:
        isnumpy = True
    else:
        isnumpy = False
    Itype = I.dtype
    # for int, I need to convert to float for interpolation
    AphiI = interp(x,torch.as_tensor(I.astype(float),dtype=Xout.dtype,device=Xout.device),Xout,mode='nearest').cpu()
    if isnumpy:
        AphiI = AphiI.numpy().astype(Itype)
    else:
        AphiI = AphiI.int()
    return AphiI
    
            

        
def rigid2D(xI,I,xJ,J,**kwargs):
    ''' 
    Rigid transformation between 2D slices.
    
    
    '''
    pass
        
        
# now we'll start building an interface
if __name__ == '__main__':
    # set up command line args
    # we will either calculate mappings or apply mappings
    parser = argparse.ArgumentParser(description='Calculate or apply mappings, or run a specified pipeline.', epilog='Enjoy')
    
    
    parser.add_argument('-m','--mode', help='Specify mode as one of register, transform, or a named pipeline', default='register',choices=['register','transform']) 
    # add other choices
    # maybe I don't need a mode
    # if I supply a -x it will apply transforms
    
    parser.add_argument('-a','--atlas',
                        help='Specify the filename of the image to be transformed (atlas)')
    parser.add_argument('-l','--label', help='Specify the filename of the label image in atlas space for QC and outputs')
    
    parser.add_argument('-t','--target', help='Specify the filename of the image to be transformed to (target)')
    parser.add_argument('-w','--weights', help='Specify the filename of the target image weights (defaults to ones)')
    
    
    parser.add_argument('-c','--config', help='Specify the filename of json config file') # only required for reg
    
    parser.add_argument('-x','--xform', help='Specify a list of transform files to apply, or a previous output directory',action='append')
    parser.add_argument('-d','--direction', help='Specify the direction of transforms to apply, either f for forward or b for backward',action='append')
    
    parser.add_argument('-o','--output', help='A directory for outputs', required=True)
    
    parser.add_argument('--output_image_format', help='File format for outputs (vtk legacy and nibabel supported)', default='.vtk')
    parser.add_argument('--num_threads', help='Optionally specify number of threads in torch', type=int)

    args = parser.parse_args()
    
    print(args)
    
    if args.num_threads is not None:
        print(f'Setting numer of torch threads to {args.num_threads}')
        torch.set_num_threads(args.num_threads)
    

    
    
    # if mode is register
    if args.mode == 'register':   
        print('Starting register pipeline')    
        if args.config is None:
            raise Exception('Config file option be set to run registration')
        
        print(f'Making output directory {args.output}')
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        print('Finished making output directory')
    
    
        # load config
        print('Loading config')
        with open(args.config) as f:
            config = json.load(f)
        # I'm getting this for initial downsampling for preprocessing
        downJs = config['downJ']
        downIs = config['downI']
        for k in config:
            print(k,config[k])
        print('Finished loading config')
        
        # load atlas
        #atlas_name = '/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-mri-reorient.vtk'
        #label_name = '/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-region_seg-reorient.vtk'
        #target_name = '/home/dtward/data/csh_data/marmoset/m1229/M1229MRI/MRI/exvivo/HR_T2/HR_T2_CM1229F-reorient.vtk'

        # TODO check works with nifti
        atlas_name = args.atlas
        if atlas_name is None:
            raise Exception('You must specify an atlas name to run the registration pipeline')
        print(f'Loading atlas {atlas_name}')
        parts = os.path.splitext(atlas_name)
        #if parts[-1] != '.vtk':
        #    raise Exception(f'Only vtk format atlas supported, but this file is {parts[-1]}')
        xI,I,title,names = read_data(atlas_name)        
        
        
        I = I.astype(float)
        # pad the first axis if necessary
        if I.ndim == 3: 
            I = I[None]
        elif I.ndim < 3 or I.ndim > 4:
            raise Exception(f'3D data required but atlas image has dimension {I.ndim}')            
        output_dir = os.path.join(args.output,'inputs/')
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        
        
        print(f'Initial downsampling so not too much gpu memory')
        # initial downsampling so there isn't so much on the gpu
        mindownI = np.min(np.array(downIs),0)
        xI,I = downsample_image_domain(xI,I,mindownI)
        downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
        dI = [x[1]-x[0] for x in xI]
        print(dI)
        nI = np.array(I.shape,dtype=int)
        # update our config variable
        config['downI'] = downIs
        
        fig = draw(I,xI)
        fig[0].suptitle('Atlas image')
        fig[0].savefig(os.path.join(output_dir,'atlas.png'))
        print('Finished loading atlas')
        
        # load target and possibly weights
        target_name = args.target
        if target_name is None:
            raise Exception('You must specify a target name to run the registration pipeline')
        print(f'Loading target {target_name}')
        parts = os.path.splitext(target_name)
        if not parts[-1]:
            print('Loading slices from directory')
            xJ,J,W0 = load_slices(target_name)            
            
        elif parts[-1] == '.vtk':
            print('Loading volume from vtk')
            xJ,J,title,names = read_vtk_data(target_name) # there's a problem here with marmoset, too many values t ounpack
            if J.ndim == 3:
                J = J[None]
            elif J.ndim < 3 or J.ndim > 4:
                raise Exception(f'3D data required but target image has dimension {J.ndim}')
            if args.weights is not None:
                print('Loading weights from vtk')
            else:
                W0 = np.ones_like(J[0])
                
        print(f'Initial downsampling so not too much gpu memory')
        # initial downsampling so there isn't so much on the gpu
        mindownJ = np.min(np.array(downJs),0)
        xJ,J = downsample_image_domain(xJ,J,mindownJ)
        W0 = downsample(W0,mindownJ)
        downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]        
        # update our config variable
        config['downJ'] = downJs
        
        # draw it
        fig = draw(J,xJ)
        fig[0].suptitle('Target image')
        fig[0].savefig(os.path.join(output_dir,'target.png'))        
        print('Finished loading target')
        
        # get one more qc file applying the initial affine
        try:
            A = np.array(config['A']).astype(float)
        except:
            A = np.eye(4)
        # this affine matrix should be 4x4, but it may be 1x4x4
        if A.ndim > 2:
            A = A[0]
        Ai = np.linalg.inv(A)
        XJ = np.stack(np.meshgrid(*xJ,indexing='ij'),-1)
        Xs = (Ai[:3,:3]@XJ[...,None])[...,0] + Ai[:3,-1]
        out = interp(xI,I,Xs.transpose((3,0,1,2)))
        fig = draw(out,xJ)
        fig[0].suptitle('Initial transformed atlas')
        fig[0].savefig(os.path.join(output_dir,'atlas_to_target_initial_affine.png'))        

        # default pipeline
        print('Starting registration pipeline')
        output = emlddmm_multiscale(I=I/np.mean(np.abs(I)),xI=[xI],J=J/np.mean(np.abs(J)),xJ=[xJ],W0=W0,**config)
        if type(output) == list:
            output = output[-1]
        print('Finished registration pipeline')
        
        # write transforms
        print('Starting to write transforms')
        write_transform_outputs(args.output,output)
        print('Finished writing transforms')
        
        # write qc outputs
        # this requires a segmentation image
        if args.label is not None:
            print('Starting to read label image for qc')
            xS,S,title,names = read_data(args.label)            
            S = S.astype(np.int32) # with int32 should be supported by torch
            print('Finished reading label image')
            print('Starting to write qc outputs')
            write_qc_outputs(args.output,output,xI,I,xJ,J,xS=xI,S=S)
            print('Finished writing qc outputs')
            
            
        # transform imaging data
        # transform target back to atlas
        Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))
        Xout = compose_sequence(args.output,Xin)
        Jt = apply_transform_float(xJ,J,Xout)
        
        # transform atlas to target
        Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xJ]))
        Xout = compose_sequence(args.output,Xin,direction='b')
        It = apply_transform_float(xI,I,Xout)
        if args.label is not None:
            St = apply_transform_int(xS,S,Xout)
        # write
        ext = args.output_image_format
        if ext[0] != '.': ext = '.' + ext
            
        atlas_output_dir = os.path.join(args.output,'to_atlas')
        if not os.path.isdir(atlas_output_dir): os.mkdir(atlas_output_dir)
        target_output_dir = os.path.join(args.output,'to_target')
        if not os.path.isdir(target_output_dir): os.mkdir(target_output_dir)                
        # output data in atlas space, use xI for voxel spacing
        write_data(os.path.join(atlas_output_dir,'target_to_atlas'+ext),xI,Jt,'target_to_atlas')

        # output data in target space, use xJ for voxel spacing
        write_data(os.path.join(target_output_dir,'atlas_to_target'+ext),xJ,It,'atlas_to_target')
        write_data(os.path.join(target_output_dir,'atlas_seg_to_target'+ext),xJ,St,'atlas_seg_to_target')


        print('Finished registration pipeline')
    elif args.mode == 'transform':
        
        # now to apply transforms, every one needs a f or a b
        # some preprocessing
        if args.direction is None:
            args.direction = ['f']
        if len(args.direction) == 1:
            args.direction = args.direction * len(args.xform)        
        if len(args.xform) != len(args.direction):
            raise Exception(f'You must input a direction for each transform, but you input {len(args.xform)} transforms and {len(args.direction)} directions')
        for tform,direction in zip(args.xform,args.direction):
            if direction.lower() not in ['f','b']:
                raise Exception(f'Transform directions must be f or b, but you input {direction}')
            
            
            
        # load atlas
        # load target (to get space info)
        # compose sequence of transforms
        # transform the data
        # write it out
        if args.atlas is not None:
            atlas_name = args.atlas
        elif args.label is not None:
            atlas_name = args.label
        else:
            raise Exception('You must specify an atlas name or label name to run the transformation pipeline')
        print(f'Loading atlas {atlas_name}')
        parts = os.path.splitext(atlas_name)        
        xI,I,title,names = read_data(atlas_name)        
        
        # load target and possibly weights
        target_name = args.target
        if target_name is None:
            raise Exception('You must specify a target name to run the transformation pipeline (TODO, support specifying a domain rather than an image)')
        print(f'Loading target {target_name}')
        parts = os.path.splitext(target_name)
        xJ,J,title,names = read_data(target_name)
        
        # to transform an image we start with Xin, and compute Xout
        # Xin will be the grid of points in target
        Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xJ]))
        Xout = compose_sequence([(x,d) for x,d  in zip(args.xform,args.direction) ], Xin)
        if args.atlas is not None:
            It = apply_transform_float(xI,I,Xout)
        else:
            It = apply_transform_int(xI,I,Xout)
        
        # write out the outputs        
        ext = args.output_image_format
        if ext[0] != '.': ext = '.' + ext                    
        if not os.path.isdir(args.output): os.mkdir(args.output)
        # name should be atlas name to target name, but without the path
        name = os.path.splitext(os.path.split(atlas_name)[1])[0] + '_to_' + os.path.splitext(os.path.split(target_name)[1])[0]
        write_data(os.path.join(args.output,name+ext),xJ,It,'transformed data')
        # write a text file that summarizes this
        name = os.path.join(args.output,name+'.txt')
        with open(name,'wt') as f:
            f.write(str(args))
         
    
    # also 
