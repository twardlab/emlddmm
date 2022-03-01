# %%
import emlddmm
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import sys
# import getopt
import argparse
import glob

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
dtype = torch.float

# %%
# utility function to form edge between two vertices
# source and dest
def add_edge(adj, spaces, src_space, src_img, dest_space, out):
 
    adj[spaces[src_space]][spaces[dest_space]] = (os.path.join(out,f'{dest_space}/{src_space}_to_{dest_space}/'), 'f')
    adj[spaces[dest_space]][spaces[src_space]] = (os.path.join(out,f'{dest_space}/{src_space}_to_{dest_space}/'), 'b')


# a modified version of BFS that stores predecessor
# of each vertex in array p
# and its distance from source in array d
def BFS(adj, src, dest, v, pred, dist):
 
    # a queue to maintain queue of vertices whose
    # adjacency list is to be scanned as per normal
    # DFS algorithm
    queue = []
  
    # boolean array visited[] which stores the
    # information whether ith vertex is reached
    # at least once in the Breadth first search
    visited = [False for i in range(v)]
  
    # initially all vertices are unvisited
    # so v[i] for all i is false
    # and as no path is yet constructed
    # dist[i] for all i set to infinity
    for i in range(v):
 
        dist[i] = 1000000
        pred[i] = -1
     
    # now source is first to be visited and
    # distance from source to itself should be 0
    visited[src] = True
    dist[src] = 0
    queue.append(src)
  
    # standard BFS algorithm
    while (len(queue) != 0):
        u = queue[0]
        queue.pop(0)
        for i in range(len(adj[u])):
         
            if (visited[list(adj[u])[i]] == False):
                visited[list(adj[u])[i]] = True
                dist[list(adj[u])[i]] = dist[u] + 1
                pred[list(adj[u])[i]] = u
                queue.append(list(adj[u])[i])
  
                # We stop BFS when we find
                # destination.
                if (list(adj[u])[i] == dest):
                    return True
  
    return False
  
  
# function to print the shortest distance
# between source vertex and destination vertex
def findShortestPath(adj, src, dest, v):
     
    # predecessor[i] array stores predecessor of
    # i and distance array stores distance of i
    # from s
    pred=[0 for i in range(v)]
    dist=[0 for i in range(v)]
  
    if (BFS(adj, src, dest, v, pred, dist) == False):
        print("Given source and destination are not connected")
  
    # vector path stores the shortest path
    path = []
    crawl = dest
    path.append(crawl)
     
    while (pred[crawl] != -1):
        path.append(pred[crawl])
        crawl = pred[crawl]
     
    path.reverse()

    # distance from source is in distance array
    print("Shortest path length is: " + str(dist[dest]), end = '')

    return path


def getTransformation(adj, path):
    transformation = []
    for i in range(len(path)-1):
        transformation.append(adj[path[i]][path[i+1]])

    return transformation


def reg(dest, source, registration, config, out, labels=None):
    src_path = source
    src_space = registration[0][0]
    src_img = registration[0][1]
    dest_path = dest
    dest_img = registration[1][1]
    dest_space = registration[1][0]
    config_file = config
    output_dir = out #os.path.join(out, '{dest}'.format(dest_space))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_name = labels
    with open(config_file) as f:
        config = json.load(f)
        f.close()
    # I'm getting this for initial downsampling for preprocessing
    downIs = config['downI']
    downJs = config['downJ']

    # atlas
    xI,I,title,names = emlddmm.read_data(dest_path)
    I = I.astype(float)
    # normalize
    I /= np.mean(np.abs(I))
    I /= np.quantile(I, 0.99)

    # initial downsampling so there isn't so much on the gpu
    mindownI = np.min(np.array(downIs),0)
    xI,I = emlddmm.downsample_image_domain(xI,I,mindownI)
    downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
    # update our config variable
    config['downI'] = downIs

    # target
    xJ,J,title,names = emlddmm.read_data(src_path)
    if 'mask' in names:
        maskind = names.index('mask')
        W0 = J[maskind]
        J = J[np.arange(J.shape[0])!=maskind]
    else:
        W0 = np.ones_like(J[0])
    J = J.astype(float)
    # normalize
    J /= np.mean(np.abs(J))
    normJ = np.quantile(J, 0.99)
    J /= normJ

    # initial downsampling so there isn't so much on the gpu
    mindownJ = np.min(np.array(downJs),0)
    xJ,J = emlddmm.downsample_image_domain(xJ,J,mindownJ)
    W0 = emlddmm.downsample(W0,mindownJ)
    downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
    # update our config variable
    config['downJ'] = downJs

    if 'A' in config:
        A = np.array(config['A']).astype(float)
    else:
        A = np.eye(4)
    print(A)

    # if 'slice_matching' not in config:
    #     # for simplicity I will add a translation manually
    #     A[:3, -1] = [-4000, 100, 4000]
    #     config['A'] = A

    # update sigma factors based on normalization
    if 'sigmaR' in config:
        config['sigmaR'][0] /= normJ
    if 'sigmaA' in config:
        config['sigmaA'][0] /= normJ
    if 'sigmaB' in config:
        config['sigmaB'][0] /= normJ
    if 'sigmaM' in config:
        config['sigmaM'][0] /= normJ

    device = 'cuda:0'
    # device = 'cpu'
    output = emlddmm.emlddmm_multiscale(I=I,xI=[xI],J=J,xJ=[xJ],W0=W0,device=device,full_outputs=False,**config)
    #write outputs
    print('saving transformations to ' + output_dir + '...')
    emlddmm.write_transform_outputs(output_dir, src_space, dest_space, xJ, xI, output[-1], src_path=src_path)
    print('saving qc to ' + output_dir + '...')
    if label_name:
        # get labels
        xS,S,title,names = emlddmm.read_data(label_name,endian='l')
        emlddmm.write_qc_outputs(output_dir, src_space, src_img, dest_space, dest_img, output[-1],xI,I,xJ,J,xS=xS,S=S.astype(float))
    else:
        emlddmm.write_qc_outputs(output_dir, src_space, src_img, dest_space, dest_img, output[-1],xI,I,xJ,J)

    return


def run_registrations(reg_list):
    # input: list of dicts of sources, targets, labels, configs, and the output dir 
    # Return: adj list, dict of space name keys and node number values

    # perform registration using reg_list as input
    for r in reg_list:
        dest = r['dest']
        source = r['source']
        registration = r['registration']
        config = r['config']
        out = r['output']
        if 'label_name' in r:
            label_name = r['label_name']
            reg(dest, source, registration, config, out, labels=label_name)
        else:
            reg(dest, source, registration, config, out)
         
    # construct spaces dict
    spaces = {}
    v = 0
    for i in reg_list:
        for j in [i['registration'][0][0], i['registration'][1][0]]: # for space names in registration
            if j not in spaces:
                spaces[j] = v
                v += 1

    adj = [{} for i in range(len(spaces))]

    for i in range(len(reg_list)):
        src_space = reg_list[i]['registration'][0][0]
        src_img = reg_list[i]['registration'][0][1]
        dest_space = reg_list[i]['registration'][1][0]
        out = reg_list[i]['output']
        add_edge(adj, spaces, src_space, src_img, dest_space, out)

    return adj, spaces


# Do new transform using composition of calculated transformations
def do_transformation(adj, spaces, src_space, src_img, dest_space, out, src_path='', dest_path=''):
    # input: image to be transformed (src_path or I), img space to to which the source image will be matched (dest_path, J), adjacency list and spaces dict from run_registration, source and destination space names
    # return: transfromed image
    # TODO: there may be an error if both src and dest images are histology series.
    
    # load source image
    xJ, J, J_title, _ = emlddmm.read_data(src_path) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    xJ = [torch.as_tensor(x,dtype=dtype,device=device) for x in xJ]

    '''in the special case of transforming an image series to the same space, e.g. [[HIST, myelin], [HIST, nissl]], we output HIST_INPUT_to_HIST_REGISTERED images
    and HIST_REGISTERED_to_HIST_INPUT images'''
    if J_title == 'slice_dataset' and dest_space == src_space:
        # get image slice names for naming output images
        dest_slice_names = [os.path.splitext(x)[0] for x in os.listdir(dest_path) if x[-4:] == 'json']
        dest_slice_names = sorted(dest_slice_names, key=lambda x: x[-4:])
        src_slice_names = [os.path.splitext(x)[0] for x in os.listdir(src_path) if x[-4:] == 'json']
        dest_slice_names = sorted(src_slice_names, key=lambda x: x[-4:])

        # first save out registered_to_input images
        # TODO: this is just the original images. The image names don't make much sense in this case.
        img_out = os.path.join(out, f'{space}/{space}_REGISTERED_to_{space}_INPUT/images')
        if not os.path.exists(img_out):
            os.makedirs(img_out)
        for i in range(J.shape[1]):
            J_ = J[:, i, None, ...]
            xJ_ = [torch.tensor([xJ[0][i], xJ[0][i]+10]), xJ[1], xJ[2]]
            title = f'{space}_REGISTERED_{src_slice_names[i]}_to_{space}_INPUT_{dest_slice_names[i]}'
            emlddmm.write_vtk_data(os.path.join(img_out, f'{space}_REGISTERED_{src_slice_names[i]}_to_{space}_INPUT_{dest_slice_names[i]}.vtk'), xJ_, J_, title)

        x_series = xJ
        X_series = torch.stack(torch.meshgrid(x_series), -1)
        space = space
        transforms = os.path.join(out, f'{space}/{space}_INPUT_to_{space}_REGISTERED/transforms')
        transforms_ls = sorted(os.listdir(transforms), key=lambda x: x.split('_matrix.txt')[0][-4:])

        A2d = []
        for t in transforms_ls:
            A2d_ = np.genfromtxt(os.path.join(transforms, t), delimiter=',')
            # note that there are nans at the end if I have commas at the end
            if np.isnan(A2d_[0, -1]):
                A2d_ = A2d_[:, :A2d_.shape[1] - 1]
            A2d.append(A2d_)

        A2d = torch.as_tensor(np.stack(A2d),dtype=dtype,device=device)
        A2di = torch.inverse(A2d)
        points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] 
        m0 = torch.min(points[..., 0])
        M0 = torch.max(points[..., 0])
        m1 = torch.min(points[..., 1])
        M1 = torch.max(points[..., 1])
        # construct a recon domain
        dJ = [x[1] - x[0] for x in x_series]
        # print('dJ shape: ', [x.shape for x in dJ])
        xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
        xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
        xr = x_series[0], xr0, xr1
        XR = torch.stack(torch.meshgrid(xr), -1)
        # reconstruct 2d series
        Xs = torch.clone(XR)
        Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
        Xs = Xs.permute(3, 0, 1, 2)
        Jr = emlddmm.interp(xJ, J, Xs)

        # save transformed 2d images   
        img_out = os.path.join(out, f'{space}/{space}_INPUT_to_{space}_REGISTERED/images')
        if not os.path.exists(img_out):
            os.makedirs(img_out)
        for i in range(Jr.shape[1]):
            Jr_ = Jr[:, i, None, ...]
            xr_ = [torch.tensor([xr[0][i], xr[0][i]+10]), xr[1], xr[2]]
            title = f'{space}_input_{src_slice_names[i]}_to_{space}_registered_{dest_slice_names[i]}'
            emlddmm.write_vtk_data(os.path.join(img_out, f'{space}_input_{src_slice_names[i]}_to_{space}_registered_{dest_slice_names[i]}.vtk'), xr_, Jr_, title)
        return

    # load destination image
    xI, I, I_title, _ = emlddmm.read_data(dest_path) # the space to transform into
    I = I.astype(float)
    I = torch.as_tensor(I, dtype=dtype, device=device)
    xI = [torch.as_tensor(x,dtype=dtype,device=device) for x in xI]

    slice_matching = 'slice_dataset' in [I_title, J_title]
    # if slice_matching then construct the reconstructed space XR
    if slice_matching:
        if I_title=='slice_dataset': # then the last transform in transformation_seq should contain A2d files
            # transforms = os.path.join(transformation_seq[-1][0], 'transforms')
            transforms = os.path.join(out, f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED/')
        else: # otherwise the first transform in transformation_seq should contain A2d filess
            # transforms = os.path.join(transformation_seq[0][0], 'transforms')
            transforms = os.path.join(out, f'{src_space}_REGISTERED/{src_space}_INPUT_to_{src_space}_REGISTERED/') 
        transforms_ls = [f for f in os.listdir(transforms) if 'vtk' not in f and 'A.txt' not in f]
        transforms_ls = sorted(transforms_ls, key=lambda x: x.split('_matrix.txt')[0][-4:])
        # determine which image is constructed from a 2d series, I or J.
        x_series = xI if I_title=='slice_dataset' else xJ
        X_series = torch.stack(torch.meshgrid(x_series),-1)

        A2d = []
        for t in transforms_ls:
            A2d_ = np.genfromtxt(os.path.join(transforms, t), delimiter=',')
            # note that there are nans at the end if I have commas at the end
            if np.isnan(A2d_[0, -1]):
                A2d_ = A2d_[:, :A2d_.shape[1] - 1]
            A2d.append(A2d_)

        A2d = torch.as_tensor(np.stack(A2d),dtype=dtype,device=device)
        A2di = torch.inverse(A2d)
        points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] # reconstructed space needs to be created from the 2d series coordinates
        m0 = torch.min(points[..., 0])
        M0 = torch.max(points[..., 0])
        m1 = torch.min(points[..., 1])
        M1 = torch.max(points[..., 1])
        # construct a recon domain
        dJ = [x[1] - x[0] for x in x_series]
        # print('dJ shape: ', [x.shape for x in dJ])
        xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
        xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
        xr = x_series[0], xr0, xr1
        XR = torch.stack(torch.meshgrid(xr), -1)
        # reconstruct 2d series
        Xs = torch.clone(XR)
        Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
        Xs = Xs.permute(3, 0, 1, 2)

    path = findShortestPath(adj, spaces[src_space], spaces[dest_space], len(spaces))
    if len(path) < 2:
        return
    print("\nPath is:")

    # printing path as sequence of space names
    for i in path:
        for key, value in spaces.items():
            if i == value:
                print(key, end=' ')

    transformation_seq = getTransformation(adj, path)
    print('\nTransformation sequence: ', transformation_seq)

    # if slice_matching and the destination is 2d series, then X = XR
    if I_title == 'slice_dataset':
        X = torch.clone(XR.permute(3,0,1,2)) # the reconstructed registered domain
        # we will also need the input domain for getting to_input displacement and images later
        X_series_ = torch.clone(X_series)            
        X_series_[1:] = ((A2di[:,None,None,:2,:2]@ (X_series[1:].permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)
        for i in reversed(range(len(transformation_seq))):
            Xin = emlddmm.compose_sequence([transformation_seq[i]], Xin)
    else:
        X = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))
    for i in reversed(range(len(transformation_seq))):
        X = emlddmm.compose_sequence([transformation_seq[i]], X)

    # get displacement
    if I_title == 'slice_dataset':
        # for input disp we need to apply compose_sequence to  A2di @ X_series to get Xin and then input_disp = Xin - X_series
        input_disp = (Xin - X_series.permute(3,0,1,2).cpu())[None] #TODO: check on this, I'm still not sure.
        registered_disp = (X - XR.permute(3,0,1,2).cpu())[None]
        
        # save out displacement from input and from registered space
        input_dir = os.path.join(out, f'{dest_space}_INPUT/{src_space}_to_{dest_space}_INPUT/transforms/')
        if not os.path.isdir(input_dir):
            os.makedirs(input_dir)
        
        registered_dir = os.path.join(out, f'{dest_space}_REGISTERED/{src_space}_to_{dest_space}_REGISTERED/transforms/')
        if not os.path.isdir(registered_dir):
            os.makedirs(registered_dir)

        # get image names of slices for naming outputs
        slice_names = [os.path.splitext(x)[0] for x in os.listdir(dest_path) if x[-4:] == 'json']
        slice_names = sorted(slice_names, key=lambda x: x[-4:])

        for i in range(input_disp.shape[2]):
            x_series_ = [x_series[0][i], x_series[1], x_series[2]]
            x_series_[0] = torch.tensor([x_series_[0], x_series_[0] + 10])
            xr_ = [xr[0][i], xr[1], xr[2]]
            xr_[0] = torch.tensor([xr_[0], xr_[0] + 10])
            # write out input to dest displacement
            output_name = os.path.join(input_dir, f'{dest_space}_INPUT_{slice_names[i]}_to_{src_space}_displacement.vtk')
            title = f'{dest_space}_INPUT_{slice_names[i]}_to_{src_space}_displacement'
            emlddmm.write_vtk_data(output_name, x_series_, input_disp[:,:, i, None, ...], title)
            # write out registered to dest displacement
            output_name = os.path.join(registered_dir, f'{dest_space}_REGISTERED_{slice_names[i]}_to_{src_space}_displacement.vtk')
            title = f'{dest_space}_REGISTERED_{slice_names[i]}_to_{src_space}_displacement'
            emlddmm.write_vtk_data(output_name, xr_, registered_disp[:,:, i, None, ...], title)

            # TODO: write out detjac for 2d displacements

    else:
        # save out 3d displacement
        disp = (X - torch.stack(torch.meshgrid(xI)).to('cpu'))[None]
        
        if J_title == 'slice_dataset':
            transform_dir = os.path.join(out, f'{dest_space}/{src_space}_REGISTERED_to_{dest_space}/transforms/')
            if not os.path.exists(transform_dir):
                os.makedirs(transform_dir)
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_REGISTERED_displacement.vtk')
            title = f'{dest_space}_to_{src_space}_REGISTERED_displacement'

        else:
            transform_dir = os.path.join(out, f'{dest_space}/{src_space}_to_{dest_space}/transforms/')
            if not os.path.isdir(transform_dir):
                os.makedirs(transform_dir)  
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_displacement.vtk')
            title = f'{dest_space}_to_{src_space}_displacement'

        emlddmm.write_vtk_data(output_name, xI, disp, title)

        # write out determinant of jacobian (detjac) of displacement
        dv = [(x[1]-x[0]).to('cpu') for x in xI]
        grad = np.stack(np.gradient(disp[0], 1, dv[0], dv[1], dv[2]), axis=-1)
        grad = np.reshape(grad, grad.shape[:-1]+(2,2))
        detjac = np.linalg.det(grad)
        if J_title == 'slice_dataset':
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_REGISTERED_detjac.vtk')
            title = f'{dest_space}_to_{src_space}_REGISTERED_detjac'
        else:
            output_name = os.path.join(transform_dir, f'{dest_space}_to_{src_space}_detjac.vtk')
            title = f'{dest_space}_to_{src_space}_detjac'
        emlddmm.write_vtk_data(output_name, xI, detjac, title)

    # now apply transformation to image
    # if slice_matching and the source is 2d series, then use xr and Jr
    if J_title == 'slice_dataset':
        # register 2d series
        Jr = emlddmm.interp(xJ, J, Xs)
        # apply transform to registered image
        AphiI = emlddmm.apply_transform_float(xr, Jr, X.to(device))
    else:
        AphiI = emlddmm.apply_transform_float(xJ, J, X.to(device))
    
    # visualize
    if I_title == 'slice_dataset': # if the destination is 2d series
        x = xr
    else:
        x = xI
    fig = emlddmm.draw(AphiI, x)
    fig[0].suptitle(f'transformed {src_space} {src_img} to {dest_space}')
    fig[0].canvas.draw()
    plt.show()

    # save transformed images
    if I_title == 'slice_dataset':
        # first save out images of src space to registered slices
        registered_out = os.path.join(out, f'{dest_space}_REGISTERED/{src_space}_to_{dest_space}_REGISTERED/images/')
        if not os.path.exists(registered_out):
            os.makedirs(registered_out)
        for i in range(AphiI.shape[1]):
            AphiI_ = AphiI[:, i, None, ...]
            x_ = [torch.tensor([x[0][i], x[0][i]+10]), x[1], x[2]]
            emlddmm.write_vtk_data(os.path.join(registered_out, f'{src_space}_{src_img}_to_{dest_space}_registered_{slice_names[i]}.vtk'), x_, AphiI_, f'{src_space}_{src_img}_to_{dest_space}_registered_{slice_names[i]}')

        # now save images of src space to input slices
        AphiI_to_input = emlddmm.apply_transform_float(xJ, J, Xin.to(device))
        input_out = os.path.join(out, f'{dest_space}_INPUT/{src_space}_to_{dest_space}_INPUT/images/')
        if not os.path.exists(input_out):
            os.makedirs(input_out)
        for i in range(AphiI_to_input.shape[1]):
            AphiI_to_input_ = AphiI_to_input[:, i, None, ...]
            xJ_ = [torch.tensor([xJ[0][i], xJ[0][i]+10]), xJ[1], xJ[2]]
            emlddmm.write_vtk_data(os.path.join(input_out, f'{src_space}_{src_img}_to_{dest_space}_input_{slice_names[i]}.vtk'), xJ_, AphiI_to_input_, f'{src_space}_{src_img}_to_{dest_space}_input_{slice_names[i]}')
        
    else:
        if J_title == 'slice_dataset':
            img_out = os.path.join(out, f'{dest_space}/{src_space}_INPUT_to_{dest_space}/images/')
            if not os.path.exists(img_out):
                os.makedirs(img_out)

        else:
            img_out = os.path.join(out, f'{dest_space}/{src_space}_to_{dest_space}/images/')
            if not os.path.exists(img_out):
                os.makedirs(img_out)
        emlddmm.write_vtk_data(os.path.join(img_out, f'{src_space}_{src_img}_to_{dest_space}.vtk'), x, AphiI, f'{src_space}_{src_img}_to_{dest_space}')

    # save text file of transformation order
    if I_title == "slice_dataset":
        # save transformation sequence to INPUT directory
        with open(os.path.join(out, f'{dest_space}_INPUT/{src_space}_to_{dest_space}_INPUT/{src_space}_to_{dest_space}_transform_seq.txt'), 'w') as f:
            for transform in reversed(transformation_seq[1:]):
                f.write(str(transform) + ', ')
            f.write(str(transformation_seq[0]))
        # save transformation sequence to RECONSTRUCTED directory
            with open(os.path.join(out, f'{dest_space}_REGISTERED/{src_space}_to_{dest_space}_REGISTERED/{src_space}_to_{dest_space}_transform_seq.txt'), 'w') as f:
                for transform in reversed(transformation_seq[1:]):
                    f.write(str(transform) + ', ')
                f.write(str(transformation_seq[0]))

    else:
        with open(os.path.join(out, f'{dest_space}/{src_space}_to_{dest_space}/{src_space}_{src_img}_to_{dest_space}_transform_seq.txt'), 'w') as f:
            for transform in reversed(transformation_seq[1:]):
                f.write(str(transform) + ', ')
            f.write(str(transformation_seq[0]))

    return


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', nargs=1,
                        help='JSON file to be processed',
                        type=argparse.FileType('r'))
    arguments = parser.parse_args()

    input_dict = json.load(arguments.infile[0])

    output = input_dict["output"] if "output" in input_dict else os.getcwd()
    try:
        space_image_path = input_dict["space_image_path"]
    except KeyError:
        print("space_image_path is a required argument. It is a list of images, with each image being a list of the format: [\"space name\", \"image name\", \"image path\"]")
    if "registrations" in input_dict:
        registrations = input_dict["registrations"]
        try:
            configs = input_dict["configs"] # if registrations then there must also be configs
        except KeyError:
            ("configs must be included in input with registrations. configs is a list of full paths to JSON registration configuration files.")
    if "transforms" in input_dict:
        transforms = input_dict["transforms"]
    if "adj" in input_dict:
        adj = input_dict["adj"]
        try:
            spaces = input_dict["spaces"] # if adj then there must also be spaces
        except KeyError:
            ("spaces must be included with adj. spaces is a dictionary of format \{\"spacename\": <space number>\, ...}, where the space number corresponds to the space number in the adjacency list.")
    transform_all = input_dict["transform_all"] if "transform_all" in input_dict else False
    transform_all = True if transform_all == "True" or "true" else False

    # convert space_image_path to dictionary of dictionaries. (image_name-path key-values in a dict of space-img key-values)
    sip = {} # space-image-path dictionary
    for i in range(len(space_image_path)):
        if not space_image_path[i][0] in sip:
            sip[space_image_path[i][0]] = {}
        new_img = {space_image_path[i][1]: space_image_path[i][2]}
        sip[space_image_path[i][0]].update(new_img)

    print('output: ', output)
    print('space-name-path dict: ', sip)
    if registrations:
        print('registrations: ', registrations)
        print('configs: ', configs)

        reg_list = []
        for i in range(len(registrations)):
            src_space = registrations[i][0][0]
            src_img = registrations[i][0][1]
            dest_space = registrations[i][1][0]
            dest_img = registrations[i][1][1]
            src_path = sip[src_space][src_img]
            dest_path = sip[dest_space][dest_img]

            reg_list.append({'registration': registrations[i], #[src_space, dest_space],
                            'source': src_path,
                            'dest': dest_path,
                            'config': configs[i],
                            'output': output})

        print('registration list: ', reg_list, '\n')
        print('running registrations...')
        adj, spaces = run_registrations(reg_list)

        # write out adjacency list and spaces dict
        with open(os.path.join(output, 'adjacency_list.txt'), 'w') as f:
            f.write(str(adj))
        with open(os.path.join(output, 'spaces_dict.txt'), 'w') as f:
            f.write(str(spaces))

    if transform_all: # do every transfrom in both directions and transform all series images to registered space.
        transforms = []
        for i in sip.keys(): # for each space
            if os.path.isdir(sip[i][list(sip[i])[0]]): # if the path is a directory, then this is an image series 
                for r in range(len(registrations)): # get the image series that was used for registration. 
                    if registrations[r][0][1] in sip[i].keys():
                        dest_img = registrations[r][0][1] 
                for k in sip[i].keys(): # for each image series in the space,
                    transforms.append([[i, k], [i, dest_img]]) # add transform to registered space
            for j in sip.keys(): # for every other space
                if j == i:
                    continue
                dest_img = list(sip[j].keys())[0] # we just need any image in destination space so use the first.
                for k in sip[i].keys(): # for each image in current space
                    transforms.append([[i, k], [j, dest_img]])

    if transforms:
        print('transforms are: ')
        for i in transforms:
            print(str(i))
        for trans in transforms:
            src_space = trans[0][0]
            src_img = trans[0][1]
            src_path = sip[src_space][src_img]
            dest_space = trans[1][0]
            dest_img = trans[1][1]
            dest_path = sip[dest_space][dest_img] 
            do_transformation(adj, spaces, src_space, src_img, dest_space, output,\
                                    src_path=src_path, dest_path=dest_path)



#%%
if __name__ == "__main__":
    main(sys.argv[1:])