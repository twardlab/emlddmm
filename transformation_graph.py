# %%
import emlddmm
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
dtype = torch.float

# %%
# utility function to form edge between two vertices
# source and dest
def add_edge(adj, src, dest, transform):
 
    adj[src][dest] = (transform, 'f')
    adj[dest][src] = (transform, 'b')


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


def reg(dest, source, config, out, labels=None):
    dest_name = dest
    src_name = source
    config_file = config
    output_dir = out
    label_name = labels
    with open(config_file) as f:
        config = json.load(f)
        f.close()
    # I'm getting this for initial downsampling for preprocessing
    downIs = config['downI']
    downJs = config['downJ']

    # atlas
    xI,I,title,names = emlddmm.read_data(dest_name)
    I = I.astype(float)
    # normalize
    I /= np.mean(np.abs(I))

    # initial downsampling so there isn't so much on the gpu
    mindownI = np.min(np.array(downIs),0)
    xI,I = emlddmm.downsample_image_domain(xI,I,mindownI)
    downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
    # update our config variable
    config['downI'] = downIs

    # target
    xJ,J,title,names = emlddmm.read_data(src_name)
    if 'mask' in names:
        maskind = names.index('mask')
        W0 = J[maskind]
        J = J[np.arange(J.shape[0])!=maskind]
    else:
        W0 = np.ones_like(J[0])
    J = J.astype(float)
    J /= np.mean(np.abs(J))

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

    device = 'cuda:0'
    #device = 'cpu'
    output = emlddmm.emlddmm_multiscale(I=I,xI=[xI],J=J,xJ=[xJ],W0=W0,device=device,**config)
    #write outputs
    emlddmm.write_transform_outputs(output_dir,output[-1])
    if label_name:
        # get labels
        xS,S,title,names = emlddmm.read_data(label_name,endian='l')
        emlddmm.write_qc_outputs(output_dir,output[-1],xI,I,xJ,J,xS=xS,S=S.astype(float))
    else:
        emlddmm.write_qc_outputs(output_dir,output[-1],xI,I,xJ,J)

    return


def run_registrations(reg_list):
    # input: list of dicts of sources, targets, labels, configs, and the output dir 
    # Return: adj list, dict of space name keys and node number values

    # perform registration using reg_list as input
    for r in reg_list:
        dest = r['dest']
        source = r['source']
        config = r['config']
        out = r['output_dir']
        if 'label_name' in r:
            label_name = r['label_name']
            reg(dest, source, config, out, labels=label_name)
        else:
            reg(dest, source, config, out)
         
    # construct spaces dict
    spaces = {}
    v = 0
    for i in reg_list:
        for j in i['spacenames']:
            if j not in spaces:
                spaces[j] = v
                v += 1

    adj = [{} for i in range(len(spaces))]

    for i in range(len(reg_list)):
        src = reg_list[i]['spacenames'][0]
        dest = reg_list[i]['spacenames'][1]
        out = reg_list[i]['output_dir']
        add_edge(adj, spaces[src], spaces[dest], out)

    return adj, spaces


# Do new transform using composition of calculated transformations
def do_transformation(adj, spaces, src, dest, src_img='', dest_img=''):
    # input: image to be transformed (src_img or I), img space to to which the source image will be matched (dest_img, J), adjacency list and spaces dict from run_registration, source and destination space names
    # return: transfromed image

    path = findShortestPath(adj, spaces[src], spaces[dest], len(spaces))
    print("\nPath is:")

    for i in path:
        for key, value in spaces.items():
            if i == value:
                print(key, end=' ')

    transformation_seq = getTransformation(adj, path)
    print('\nTransformation sequence: ', transformation_seq)

    xI, I, I_title, _ = emlddmm.read_data(dest_img) # the space to transform into
    I = I.astype(float)
    I = torch.as_tensor(I, dtype=dtype, device=device)
    xI = [torch.as_tensor(x,dtype=dtype,device=device) for x in xI]
    xJ, J, J_title, _ = emlddmm.read_data(src_img) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    xJ = [torch.as_tensor(x,dtype=dtype,device=device) for x in xJ]

    slice_matching = 'slice_dataset' in [I_title, J_title]
    # if slice_matching then construct the reconstructed space XR
    if slice_matching:
        if I_title=='slice_dataset': # then the last transform in transformation_seq should contain A2d files
            transforms = os.path.join(transformation_seq[0][0], 'transforms')
            transforms_ls = os.listdir(transforms)
        else: # otherwise the first transform in transformation_seq should contain A2d files
            transforms = os.path.join(transformation_seq[-1][0], 'transforms')
            transforms_ls = os.listdir(transforms)
        # determine which image is constructed from a 2d series, I or J.
        x_series = xI if I_title=='slice_dataset' else xJ
        X_series = torch.stack(torch.meshgrid(x_series),-1)
        transforms_ls.pop(0)
        transforms_ls.pop(-1)

        A2d = []
        for t in transforms_ls:
            A2d_ = np.genfromtxt(os.path.join(transforms, t), delimiter=',')
            # note that there are nans at the end if I have commas at the end
            if np.isnan(A2d_[0, -1]):
                A2d_ = A2d_[:, :A2d_.shape[1] - 1]
            A2d.append(A2d_)

        A2d = torch.as_tensor(A2d,dtype=dtype,device=device)
        A2di = torch.inverse(A2d)
        points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] # reconstructed space needs to be created from the 2d series coordinates
        m0 = torch.min(points[..., 0])
        M0 = torch.max(points[..., 0])
        m1 = torch.min(points[..., 1])
        M1 = torch.max(points[..., 1])

        # construct a recon domain
        dJ = [x[1] - x[0] for x in x_series]
        xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
        xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
        xr = x_series[0], xr0, xr1
        XR = torch.stack(torch.meshgrid(xr), -1)
        # reconstruct 2d series
        Xs = torch.clone(XR)
        Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
        Xs = Xs.permute(3, 0, 1, 2)
        Jr = emlddmm.interp(xJ, J, Xs)


    # if slice_matching and the destination is 2d series, then X = XR
    if slice_matching and I_title == 'slice_dataset':
        X = torch.clone(XR.permute(3,0,1,2))
    else:
        X = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))
    for i in range(len(transformation_seq)):
        X = emlddmm.compose_sequence([transformation_seq[i]], X)

    # if slice_matching and the source is 2d series, then use xr and Jr
    if slice_matching and J_title == 'slice_dataset':
        AphiI = emlddmm.apply_transform_float(xr, Jr, X.to(device))
    else:
        AphiI = emlddmm.apply_transform_float(xJ, J, X.to(device))
    
    # visualize
    if slice_matching and I_title == 'slice_dataset': # if the destination is 2d series
        fig = emlddmm.draw(AphiI,xr)
    else:
        fig = emlddmm.draw(AphiI, xI)
    fig[0].suptitle('transformed {src} to {dest}'.format(src=src, dest=dest))
    fig[0].canvas.draw()
    plt.show()

    return AphiI
# %%
# note: the forward transformation samples the source image in the destination coordinates (i.e. makes the source to look like the dest)
reg_list = [{'spacenames': ['NISSL', 'ATLAS'], # first element of spacenames is the source space and second is destination
             'source': 'C:\\Users\\BGAdmin\\emlddmm\\MD787_small_nissl',
             'dest': 'Allen_Atlas_vtk/ara_nissl_50.vtk',
             'config': 'config787small.json',
             'output_dir': 'test_output5',
             'label_name': 'Allen_Atlas_vtk/annotation_50.vtk'}]

adj, spaces = run_registrations(reg_list)

# %%
from skimage import color
src = 'ATLAS'
dest = 'NISSL'

nissl_img = 'C:\\Users\\BGAdmin\\emlddmm\\MD787_small_nissl'
atlas_img = 'Allen_Atlas_vtk/ara_nissl_50.vtk'
AphiI = do_transformation(adj=adj, spaces=spaces, src=src, dest=dest, src_img=atlas_img, dest_img=nissl_img)