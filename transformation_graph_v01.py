from re import A
from cv2 import transform
from sklearn.metrics import det_curve
import emlddmm
import numpy as np
import torch
import argparse
from argparse import RawTextHelpFormatter
import json
import os
import pickle

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
dtype = torch.float

class Graph:
    """
    graph object with nodes and edges representing image spaces and transformations between them.

    Attributes
    ----------
    nodes : dict
        Integer keys map to the space name and a grid array of the corresponding image space.
    edges: list
       list of dictionaries holding the transforms needed to map between connecting spaces.
    """

    def __init__(self, adj=[], spaces={}):
        self.adj = adj
        self.spaces = spaces
    
    def add_space(self, space_name, x=[]):
        v = len(self.spaces)
        if space_name not in self.spaces:
            self.spaces.update({space_name: [v, x]})

    def add_edge(self, transforms, src_space, target_space):
        self.adj[self.spaces[src_space][0]].update({self.spaces[target_space][0]: transforms})

    def BFS(self, src, target, v, pred, dist):
        """ Breadth first search

        a modified version of BFS that stores predecessor
        of each vertex in array pred and its distance from source in array dist

        Parameters
        ----------
        src: int
            int value given by corresponding src in spaces dict
        target: int 
            int value given by corresponding target in spaces dict
        v: int
            length of spaces dict
        pred: list of ints
            stores predecessor of vertex i at pred[i]
        dist: list of ints
            stores distance (by number of vertices) of vertex i from source vertex

        Returns
        -------
        bool
            True if a path from src to target is found and False otherwise

        """
    

        queue = []
    
        visited = [False for i in range(v)]
        # for each space we initialize the distance from src to be a large number and the predecessor to be -1
        for i in range(v):
    
            dist[i] = 1000000
            pred[i] = -1
        
        # visit source first. Distance from source to itself is 0
        visited[src] = True
        dist[src] = 0
        queue.append(src)
    
        # BFS algorithm
        while (len(queue) != 0):
            u = queue[0]
            queue.pop(0)
            for i in range(len(self.adj[u])):
            
                if (visited[list(self.adj[u])[i]] == False):
                    visited[list(self.adj[u])[i]] = True
                    dist[list(self.adj[u])[i]] = dist[u] + 1
                    pred[list(self.adj[u])[i]] = u
                    queue.append(list(self.adj[u])[i])
    
                    # We stop BFS when we find
                    # destination.
                    if (list(self.adj[u])[i] == target):
                        return True
    
        return False

    def shortest_path(self, src, target):
        """ Find Shortest Path

        Finds the shortest path between target and src in the adjacency list and prints its length

        Parameters
        ----------
        src: int 
            src value given by corresponding source in spaces dict
        target: int
            int value given by corresponding source in spaces dict

        Returns
        -------
        path : list of ints
            path from target to src using integer values of the adjacency list vertices. Integers can be converted to space names by the spaces dict.


        Example
        -------
        >>> adj = [{1: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'f')},
        ...        {2: ('outputs/CCF/MRI_to_CCF/', 'f'), 0: ('outputs/MRI/HIST_REGISTERED_to_MRI/', 'b')},
        ...        {1: ('outputs/CCF/MRI_to_CCF/', 'b')},
        ...        {}]
        >>> path = find_shortest_path(adj, 0, 2, 4)
        Shortest path length is: 2

        >>> print(path)
        [0,1,2]

        >>> path = transformation_graph.find_shortest_path(adj, 0, 3, 4)
        Given target and source are not connected

        """
        v = len(self.spaces)
        
        pred=[0 for i in range(v)] # predecessor of space i in path from target to src
        dist=[0 for i in range(v)] # distance of vertex i by number of vertices from src
    
        if (self.BFS(src, target, v, pred, dist) == False):
            print("Given target and source are not connected")
            
        # path stores the shortest path
        path = []
        crawl = target
        path.append(crawl)
        
        while (pred[crawl] != -1):
            path.append(pred[crawl])
            crawl = pred[crawl]
        
        path.reverse()

        # if len(path) > 1:
            # distance from source is in distance array
            # print(f"Shortest path length is: {dist[target]} \n")

        return path
    
    def transforms(self, path):
        transforms = []
        for i in range(len(path)-1):
            transforms =  transforms + self.adj[path[i]][path[i+1]]

        return transforms
    
    def map_points(self, src_space, transforms):
        '''map points from source space to target. If mapping to an image series, maps to the registered domain.

        Parameters
        ----------
        srs_space : str
            name of the source space
        transforms : list of emlddmm Transform objects

        Returns
        -------
        X : torch tensor
            transformed points
        '''
        xI = self.spaces[src_space][1]
        xI = [torch.as_tensor(x) for x in xI]
        XI = torch.stack(torch.meshgrid(xI, indexing='ij'))
        # series of 2d transforms can only be applied to the space on which they were computed because the size must match the number of slices.
        # to avoid an error, check if a volumetric transform is followed by a 2d series transform.
        check = False
        ids = []
        for i in range(len(transforms)):
            A2d = transforms[i].data.ndim == 3
            if A2d and check:
                ids.append(i)
            elif not A2d:
                check = True
        transforms = [j for i, j in enumerate(transforms) if i not in ids]
        X = emlddmm.compose_sequence(transforms, XI)
        return X

    def map_image(self, src_space, target_space, image, transforms):
        '''Map an image from source space to target space.

        Parameters
        ----------
        src_space : str
            name of source space
        target_space : str
            name of target space
        image : array

        Returns
        -------
        image : array
            transformed image data
        '''
        # if the last transforms are 2d series affine, then we will first apply them to the target space and resample the target image in registered space.
        # then apply the other transforms to the source space and resample the registered target image.
        A2d = []
        for t in transforms[::-1]:
            if t.data.ndim == 3:
                A2d.insert(0,t)
            else:
                break
        xJ = self.spaces[target_space][1]
        xJ = [torch.as_tensor(x) for x in xJ]
        if len(A2d) > 0:
            image = torch.as_tensor(image)
            XJ = torch.stack(torch.meshgrid(xJ, indexing='ij'))
            XR = emlddmm.compose_sequence(A2d, XJ)
            image = emlddmm.interp(xJ,image, XR)
        X = self.map_points(src_space, transforms)
        image = emlddmm.interp(xJ, image, X)

        return image


def read_graph():
    pass


def write_graph():
    pass

def graph_reconstruct(graph, out, I, J):
    ''' Apply Transformation

    Applies affine matrix and velocity field transforms to map source points to target points. Saves displacement field from source points to target points
    (i.e. difference between transformed coordinates and input coordinates), and determinant of Jacobian for 3d source spaces. Also saves transformed image in vtk format.

    Parameters
    ----------
    graph : emlddmm Graph object
    out: str
        path to registration outputs parent directory
    I : emlddmm Image   
    J : emlddmm Image

    '''
    jacobian = lambda X,dv : np.stack(np.gradient(X, dv[0],dv[1],dv[2], axis=(1,2,3))).transpose(2,3,4,0,1)

    dtype = torch.float
    device = 'cpu'
    # convert data to torch
    J.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in J.x]
    J.data = torch.as_tensor(J.data, dtype=dtype, device=device)
    I.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in I.x]
    I.data = torch.as_tensor(I.data, dtype=dtype, device=device)

    # forward transform
    path = graph.shortest_path(graph.spaces[I.space][0], graph.spaces[J.space][0]) # shortest_path transforms points from src space to target space
    transforms = graph.transforms(path)
    XI = torch.stack(torch.meshgrid(I.x, indexing='ij'))
    # fXI = emlddmm.compose_sequence(transforms, XI)
    # fJ = emlddmm.interp(J.x, J.data, fXI)
    fXI = graph.map_points(I.space, transforms)
    fJ = graph.map_image(I.space, J.space, J.data, transforms)

    # backward transform
    path = graph.shortest_path(graph.spaces[J.space][0], graph.spaces[I.space][0])
    transforms = graph.transforms(path)
    XJ = torch.stack(torch.meshgrid(J.x, indexing='ij'))
    # fXJ = emlddmm.compose_sequence(transforms, XJ)
    # fI = emlddmm.interp(I.x, I.data, fXJ)
    fXJ = graph.map_points(J.space, transforms)
    fI = graph.map_image(J.space, I.space, I.data, transforms)
    '''
    Three cases:

    1) series to series
        Save reconstructions of each slice in the other space.
        Requires applying A2d (forward transforms) to I points and resampling J, 
        then apply A2di to J points and resampling I.
    2) volume to series
        A) Save out input to registered images
            Apply A2d to I points and resample J, then save each slice.
        B) Save volume to registered images in {J.space}_registered/{I.space}_{I.name}_to_{J.space}_registered/images/,
         and volume to input images in {J.space}_input/{I.space}_{I.name}_to_{J.space}_input/images/
        C) Save series to volume image in {I.space}/{J.space}_{J.name}_input_to_{I.space}/images/
        D) Save series to volume detjac and displacement in {I.space}/{J.space}_{J.name}_registered_to_{I.space}/transforms/
    3) volume to volume 
        A) Save out I to J image in {J.space}/{I.space}_to_{J.space}/images, and J to I image in {I.space}/{J.space}_to_{I.space}/images/
        B) Save out I to J detjac and displacement in {J.space}/{I.space}_to_{J.space}/transforms/
        and J to I detjac and displacement in {I.space}/{J.space}_to_{I.space}/transforms/
    '''

    if I.title == 'slice_dataset' and J.title == 'slice_dataset':
        # series to series
        # Assumes J and I have the same space dimensions
        J_to_I_out = os.path.join(out, f'{I.space}_input/{J.space}_input_to_{I.space}_input/images/')
        if not os.path.exists(J_to_I_out):
            os.makedirs(J_to_I_out)
        I_to_J_out = os.path.join(out, f'{J.space}_input/{I.space}_input_to_{J.space}_input/images/')
        if not os.path.exists(I_to_J_out):
            os.makedirs(I_to_J_out)
        for i in range(J.data.shape[1]):
            x = [[I.x[0][i], I.x[0][i]+10], I.x[1], I.x[2]]
            # first J to I
            img = fJ[:, i, None, ...]
            title = f'{J.space}_input_{J.fnames()[i]}_to_{I.space}_input_{I.fnames()[i]}'
            emlddmm.write_vtk_data(os.path.join(J_to_I_out, f'{J.space}_input_{J.fnames()[i]}_to_{I.space}_input_{I.fnames()[i]}.vtk'), x, img, title)
            # now I to J
            img = fI[:, i, None, ...]
            title = f'{I.space}_input_{I.fnames()[i]}_to_{J.space}_input_{J.fnames()[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_J_out, f'{I.space}_input_{I.fnames()[i]}_to_{J.space}_input_{J.fnames()[i]}.vtk'), x, img, title)

    elif J.title == 'slice_dataset':
        # volume to series
        # we need I transformed to J registered space
        path = graph.shortest_path(graph.spaces[J.space][0], graph.spaces[I.space][0])
        # omit the first transform (R^-1) which takes points from input to registered
        transforms = graph.transforms(path)[1:] # TODO: what about myelin to nissl to CCF?
        phiiAiXJ = graph.map_points(J.space, transforms)
        AphiI = graph.map_image(J.space, I.space, I.data, transforms)
        # get I to J registered and I to J input displacements
        reg_disp = (phiiAiXJ - XJ)[None]
        input_disp = (fXJ - XJ)[None]
        # J input to J registered space
        path = graph.shortest_path(graph.spaces[I.space][0], graph.spaces[J.space][0])
        # get the last transform (R) which takes points from registered to input
        transforms = [graph.transforms(path)[-1]] # TODO: what if last transform takes nissl to myelin?
        RXJ = emlddmm.compose_sequence(transforms, XJ)
        RiJ = emlddmm.interp(J.x, J.data, RXJ)
        # setup output paths
        I_to_Ji_out = os.path.join(out, f'{J.space}_input/{I.space}_{I.name}_to_{J.space}_input/images/')
        if not os.path.exists(I_to_Ji_out):
            os.makedirs(I_to_Ji_out)
        I_to_Jr_out = os.path.join(out, f'{J.space}_registered/{I.space}_{I.name}_to_{J.space}_registered/images/')
        if not os.path.exists(I_to_Jr_out):
            os.makedirs(I_to_Jr_out)
        Ji_to_Jr_out = os.path.join(out, f'{J.space}_registered/{J.space}_input_to_{J.space}_registered/images/')
        if not os.path.exists(Ji_to_Jr_out):
            os.makedirs(Ji_to_Jr_out)
        reg_disp_out = os.path.join(out, f'{J.space}_registered/{I.space}_{I.name}_to_{J.space}_registered/transforms/')
        if not os.path.exists(reg_disp_out):
            os.makedirs(reg_disp_out)
        input_disp_out = os.path.join(out, f'{J.space}_input/{I.space}_{I.name}_to_{J.space}_input/transforms/')
        if not os.path.exists(input_disp_out):
            os.makedirs(input_disp_out)
        for i in range(J.data.shape[1]):
            x = [[J.x[0][i], J.x[0][i]+10], J.x[1], J.x[2]]
            # volume to input series
            # save image
            img = fI[:, i, None, ...]
            title = f'{I.space}_{I.name}_to_{J.space}_input_{J.fnames()[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_Ji_out, title + '.vtk'), x, img, title)
            # save displacement
            disp = input_disp[:, :, i, None]
            title = f'{J.space}_input_{J.fnames()[i]}_to_{I.space}_displacement'
            emlddmm.write_vtk_data(os.path.join(input_disp_out, title + '.vtk'), x, disp, title)
            # volume to registered series
            img = AphiI[:, i, None, ...]
            title = f'{I.space}_{I.name}_to_{J.space}_registered_{J.fnames()[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_Jr_out, title + '.vtk'), x, img, title)
            # save displacement
            disp = reg_disp[:, :, i, None]
            title = f'{J.space}_registered_{J.fnames()[i]}_to_{I.space}_displacement'
            emlddmm.write_vtk_data(os.path.join(reg_disp_out, title + '.vtk'), x, disp, title)
            # input to registered images
            img = RiJ[:, i, None, ...]
            title = f'{J.space}_input_{J.fnames()[i]}_to_{J.space}_registered_{J.fnames()[i]}'
            emlddmm.write_vtk_data(os.path.join(Ji_to_Jr_out, title + '.vtk'), x, img, title)

        # J to I
        img = fJ
        title = f'{J.space}_{J.name}_input_to_{I.space}'
        J_to_I_imgs = os.path.join(out, f'{I.space}/{J.space}_{J.name}_input_to_{I.space}/images/')
        if not os.path.exists(J_to_I_imgs):
            os.makedirs(J_to_I_imgs)
        emlddmm.write_vtk_data(os.path.join(J_to_I_imgs, title + '.vtk'), I.x, img, title)
        # disp
        # we need I to J registered points
        disp = (fXI - XI)[None]
        title = f'{J.space}_{J.name}_registered_to_{I.space}_displacement'
        J_to_I_transforms = os.path.join(out, f'{I.space}/{J.space}_{J.name}_registered_to_{I.space}/transforms/')
        if not os.path.exists(J_to_I_transforms):
            os.makedirs(J_to_I_transforms)
        emlddmm.write_vtk_data(os.path.join(J_to_I_transforms, title + '.vtk'), I.x, disp, title)
        # determinant of jacobian
        dv = [(x[1]-x[0]) for x in I.x]
        jac = jacobian(fXI, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{J.space}_{J.name}_registered_to_{I.space}_detjac'
        emlddmm.write_vtk_data(os.path.join(J_to_I_transforms, title + '.vtk'), I.x, detjac, title)

    else:
        # volume to volume
        # J to I
        # save image
        img = fJ
        title = f'{J.space}_{J.name}_to_{I.space}'
        J_to_I_imgs = os.path.join(out, f'{I.space}/{J.space}_{J.name}_to_{I.space}/images/')
        if not os.path.exists(J_to_I_imgs):
            os.makedirs(J_to_I_imgs)
        emlddmm.write_vtk_data(os.path.join(J_to_I_imgs, title + '.vtk'), I.x, img, title)
        # save displacement
        disp = (fXI - XI)[None]
        title = f'{J.space}_{J.name}_to_{I.space}_displacement'
        J_to_I_transforms = os.path.join(out, f'{I.space}/{J.space}_{J.name}_to_{I.space}/transforms/')
        if not os.path.exists(J_to_I_transforms):
            os.makedirs(J_to_I_transforms)
        emlddmm.write_vtk_data(os.path.join(J_to_I_transforms, title + '.vtk'), I.x, disp, title)
        # save determinant of jacobian
        dv = [(x[1]-x[0]) for x in I.x]
        jac = jacobian(fXI, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{J.space}_{J.name}_to_{I.space}_detjac'
        emlddmm.write_vtk_data(os.path.join(J_to_I_transforms, title + '.vtk'), I.x, detjac, title)

        # I to J
        # save image
        img = fI
        title = f'{I.space}_{I.name}_to_{J.space}'
        I_to_J_imgs = os.path.join(out, f'{J.space}/{I.space}_{I.name}_to_{J.space}/images/')
        if not os.path.exists(I_to_J_imgs):
            os.makedirs(I_to_J_imgs)
        emlddmm.write_vtk_data(os.path.join(I_to_J_imgs, title + '.vtk'), J.x, img, title)
        # save displacement
        disp = (fXJ - XJ)[None]
        title = f'{I.space}_{I.name}_to_{J.space}_displacement'
        I_to_J_transforms = os.path.join(out, f'{J.space}/{I.space}_{I.name}_to_{J.space}/transforms/')
        if not os.path.exists(I_to_J_transforms):
            os.makedirs(I_to_J_transforms)
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), J.x, disp, title)
        # save determinant of jacobian
        dv = [(x[1]-x[0]) for x in J.x]
        jac = jacobian(fXJ, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{I.space}_{I.name}_to_{J.space}_detjac'
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), J.x, detjac, title)

def registered_domain(x,A2d):
    '''Construct a new domain that fits all rigidly aligned slices.

    Parameters
    ----------
    x : list of arrays
        list of numpy arrays containing voxel positions along each axis.
    A2d : numpy array
        Nx3x3 array of affine transformations
    
    Returns
    -------
    xr : list of arrays
        new list of numpy arrays containing voxel positions along each axis

    '''
    X = torch.stack(torch.meshgrid(x, indexing='ij'), -1)
    A2di = torch.inverse(A2d)
    points = (A2di[:, None, None, :2, :2] @ X[..., 1:, None])[..., 0] 
    m0 = torch.min(points[..., 0])
    M0 = torch.max(points[..., 0])
    m1 = torch.min(points[..., 1])
    M1 = torch.max(points[..., 1])
    # construct a recon domain
    dJ = [x[1] - x[0] for x in x]
    # print('dJ shape: ', [x.shape for x in dJ])
    xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
    xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
    xr = x[0], xr0, xr1

    return xr

def _graph_reconstruct(graph, out, I, J):
    """ Apply Transformation

    Applies affine matrix and velocity field transforms to map source points to target points. Saves displacement field from source points to target points
    (i.e. difference between transformed coordinates and input coordinates), and determinant of Jacobian for 3d source spaces. Also saves transformed image in vtk format.

    Parameters
    ----------
    graph : emlddmm Graph
    out: str
        path to registration outputs root
    I : emlddmm Image   
    J : emlddmm Image
    
    Example
    -------
    >>> graph.adj = [{1: ('outputs/example_output/CCF/MRI_to_CCF/', 'f'), 2: ('outputs/example_output/MRI/HIST_registered_to_MRI/', 'b')},
    ... {0: ('outputs/example_output/CCF/MRI_to_CCF/', 'b')},
    ... {0: ('outputs/example_output/MRI/HIST_registered_to_MRI/', 'f')}]
    >>> graph.spaces = {'MRI': 0, 'CCF': 1, 'HIST': 2}
    >>> apply_transformation(graph.adj, graph.spaces, 'MRI', 'masked', 'CCF', 'outputs/example_output',
    ... 'HR_NIHxCSHL_50um_14T_M1_masked.vtk', 'average_template_50.vtk')
    """
    # input: image to be transformed (target_path or I), img space to which the source image will be matched (src_path i.e. J), adjacency list and spaces dict from run_registration, source and source space names
    # return: transfromed image

    src_space = I.space
    src_path = I.path
    target_space = J.space
    target_img = J.name
    target_path = J.path
    
    # load source image
    xJ, J, J_title, _ = emlddmm.read_data(target_path) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    xJ = [torch.as_tensor(np.copy(x),dtype=dtype,device=device) for x in xJ]

    # if rigidly registering histology with different contrasts, reconstruct each in the other space
    if J_title == 'slice_dataset' and I.title == 'slice_dataset':
        pass

    if J_title == 'slice_dataset' and src_space == target_space:
        # get image slice names for naming output images
        src_slice_names = [os.path.splitext(x)[0] for x in os.listdir(src_path) if x[-4:] == 'json']
        src_slice_names = sorted(src_slice_names, key=lambda x: x[-4:])
        target_slice_names = [os.path.splitext(x)[0] for x in os.listdir(target_path) if x[-4:] == 'json']
        target_slice_names = sorted(target_slice_names, key=lambda x: x[-4:])
        space = src_space # target_space = src_space = space

        x_series = xJ
        X_series = torch.stack(torch.meshgrid(x_series, indexing='ij'), -1)
        transforms = os.path.join(out, f'{space}_registered/{space}_input_to_{space}_registered/transforms')
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
        XR = torch.stack(torch.meshgrid(xr,indexing='ij'), -1)
        # reconstruct 2d series
        Xs = torch.clone(XR)
        Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
        Xs = Xs.permute(3, 0, 1, 2)
        Jr = emlddmm.interp(xJ, J, Xs)

        # save transformed 2d images   
        img_out = os.path.join(out, f'{space}_registered/{space}_input_to_{space}_registered/images')
        if not os.path.exists(img_out):
            os.makedirs(img_out)
        for i in range(Jr.shape[1]):
            Jr_ = Jr[:, i, None, ...]
            xr_ = [torch.tensor([xr[0][i], xr[0][i]+10]), xr[1], xr[2]]
            title = f'{space}_input_{target_slice_names[i]}_to_{space}_registered_{src_slice_names[i]}'
            emlddmm.write_vtk_data(os.path.join(img_out, f'{space}_input_{target_slice_names[i]}_to_{space}_registered_{src_slice_names[i]}.vtk'), xr_, Jr_, title)

        return

    # load source image
    xI, I, I_title, _ = emlddmm.read_data(src_path) # the space to transform into
    I = I.astype(float)
    I = torch.as_tensor(I, dtype=dtype, device=device)
    xI = [torch.as_tensor(np.copy(x),dtype=dtype,device=device) for x in xI]

    slice_matching = 'slice_dataset' in [I_title, J_title]
    # if slice_matching then construct the reconstructed space XR
    if slice_matching:
        if I_title=='slice_dataset': # then the last transform in transformation_seq should contain A2d files
            # transforms = os.path.join(transformation_seq[-1][0], 'transforms')
            transforms = os.path.join(out, f'{src_space}_registered/{src_space}_input_to_{src_space}_registered/transforms')
        else: # otherwise the first transform in transformation_seq should contain A2d filess
            # transforms = os.path.join(transformation_seq[0][0], 'transforms')
            transforms = os.path.join(out, f'{target_space}_registered/{target_space}_input_to_{target_space}_registered/transforms') 
        transforms_ls = [f for f in os.listdir(transforms) if 'vtk' not in f and 'A.txt' not in f]
        transforms_ls = sorted(transforms_ls, key=lambda x: x.split('_matrix.txt')[0][-4:])
        # determine which image is constructed from a 2d series, I or J.
        x_series = xI if I_title=='slice_dataset' else xJ
        X_series = torch.stack(torch.meshgrid(x_series, indexing='ij'),-1)

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
        XR = torch.stack(torch.meshgrid(xr, indexing='ij'), -1)
        # reconstruct 2d series
        Xs = torch.clone(XR)
        Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
        Xs = Xs.permute(3, 0, 1, 2)

    path = graph.find_shortest_path(graph.adj, graph.spaces[target_space][0], graph.spaces[src_space][0], len(graph.spaces))
    if len(path) < 2:
        return
    print("\nPath is:")

    # printing path as sequence of space names
    for i in path:
        for key, value in graph.spaces.items():
            if i == value[0]:
                print(key, end=' ')

    transformation_seq = graph.get_transformation(graph.adj, path)
    print('\nTransformation sequence: ', transformation_seq)

    # if slice_matching and the source is 2d series, then X = XR
    if I_title == 'slice_dataset':
        X = torch.clone(XR.permute(3,0,1,2)) # the reconstructed registered domain

        # we will also need the input domain for getting to_input displacement and images later
        Xin = torch.clone(X_series) # note that coordinates are on the last dimension e.g. (i,j,k,3)           
        Xin[..., 1:] = ((A2di[:,None,None,:2,:2] @ (Xin[..., 1:][..., None]))[...,0] + A2di[:,None,None,:2,-1])
        Xin = Xin.permute(3,0,1,2) # (3,i,j,k)
        for i in reversed(range(len(transformation_seq))):
            Xin = emlddmm.compose_sequence([transformation_seq[i]], Xin)
    else:
        X = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI], indexing='ij'))
    for i in reversed(range(len(transformation_seq))):
        X = emlddmm.compose_sequence([transformation_seq[i]], X)

    # get displacement
    if I_title == 'slice_dataset':
        # for input disp we need to apply compose_sequence to  A2di @ X_series to get Xin and then input_disp = Xin - X_series
        input_disp = (Xin - X_series.permute(3,0,1,2).cpu())[None]
        registered_disp = (X - XR.permute(3,0,1,2).cpu())[None]
        
        # save out displacement from input and from registered space
        input_dir = os.path.join(out, f'{src_space}_input/{target_space}_to_{src_space}_input/transforms/')
        if not os.path.isdir(input_dir):
            os.makedirs(input_dir)
        
        registered_dir = os.path.join(out, f'{src_space}_registered/{target_space}_to_{src_space}_registered/transforms/')
        if not os.path.isdir(registered_dir):
            os.makedirs(registered_dir)

        # get image names of slices for naming outputs
        slice_names = [os.path.splitext(x)[0] for x in os.listdir(src_path) if x[-4:] == 'json']
        slice_names = sorted(slice_names, key=lambda x: x[-4:])

        for i in range(input_disp.shape[2]):
            x_series_ = [x_series[0][i], x_series[1], x_series[2]]
            x_series_[0] = torch.tensor([x_series_[0], x_series_[0] + 10])
            xr_ = [xr[0][i], xr[1], xr[2]]
            xr_[0] = torch.tensor([xr_[0], xr_[0] + 10])
            # write out input to src displacement
            output_name = os.path.join(input_dir, f'{src_space}_input_{slice_names[i]}_to_{target_space}_displacement.vtk')
            title = f'{src_space}_input_{slice_names[i]}_to_{target_space}_displacement'
            emlddmm.write_vtk_data(output_name, x_series_, input_disp[:,:, i, None, ...], title)
            # write out registered to src displacement
            output_name = os.path.join(registered_dir, f'{src_space}_registered_{slice_names[i]}_to_{target_space}_displacement.vtk')
            title = f'{src_space}_registered_{slice_names[i]}_to_{target_space}_displacement'
            emlddmm.write_vtk_data(output_name, xr_, registered_disp[:,:, i, None, ...], title)

            # TODO: write out detjac for 2d displacements

    else:
        # save out 3d displacement
        disp = (X - torch.stack(torch.meshgrid(xI, indexing='ij')).to('cpu'))[None]
        
        if J_title == 'slice_dataset':
            transform_dir = os.path.join(out, f'{src_space}/{target_space}_registered_to_{src_space}/transforms/')
            if not os.path.exists(transform_dir):
                os.makedirs(transform_dir)
            output_name = os.path.join(transform_dir, f'{src_space}_to_{target_space}_registered_displacement.vtk')
            title = f'{src_space}_to_{target_space}_registered_displacement'

        else:
            transform_dir = os.path.join(out, f'{src_space}/{target_space}_to_{src_space}/transforms/')
            if not os.path.isdir(transform_dir):
                os.makedirs(transform_dir)  
            output_name = os.path.join(transform_dir, f'{src_space}_to_{target_space}_displacement.vtk')
            title = f'{src_space}_to_{target_space}_displacement'

        emlddmm.write_vtk_data(output_name, xI, disp, title)

        # write out determinant of jacobian (detjac) of the transformed coordinates
        dv = [(x[1]-x[0]).to('cpu') for x in xI]
        jacobian = lambda X,dv : np.stack(np.gradient(X, dv[0],dv[1],dv[2], axis=(1,2,3))).transpose(2,3,4,0,1)
        jac = jacobian(X,dv)
        detjac = np.linalg.det(jac)
        if J_title == 'slice_dataset':
            output_name = os.path.join(transform_dir, f'{src_space}_to_{target_space}_registered_detjac.vtk')
            title = f'{src_space}_to_{target_space}_registered_detjac'
        else:
            output_name = os.path.join(transform_dir, f'{src_space}_to_{target_space}_detjac.vtk')
            title = f'{src_space}_to_{target_space}_detjac'
        emlddmm.write_vtk_data(output_name, xI, detjac[None], title)

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
    if I_title == 'slice_dataset': # if the source is 2d series
        x = xr
    else:
        x = xI
    # fig = emlddmm.draw(AphiI, x)
    # fig[0].suptitle(f'transformed {target_space} {target_img} to {src_space}')
    # fig[0].canvas.draw()
    # plt.show()

    # save transformed images
    if I_title == 'slice_dataset':
        # first save out images of target space to registered slices
        registered_out = os.path.join(out, f'{src_space}_registered/{target_space}_to_{src_space}_registered/images/')
        if not os.path.exists(registered_out):
            os.makedirs(registered_out)
        for i in range(AphiI.shape[1]):
            AphiI_ = AphiI[:, i, None, ...]
            x_ = [torch.tensor([x[0][i], x[0][i]+10]), x[1], x[2]]
            emlddmm.write_vtk_data(os.path.join(registered_out, f'{target_space}_{target_img}_to_{src_space}_registered_{slice_names[i]}.vtk'), x_, AphiI_, f'{target_space}_{target_img}_to_{src_space}_registered_{slice_names[i]}')

        # now save images of target space to slices in input space
        AphiI_to_input = emlddmm.apply_transform_float(xJ, J, Xin.to(device))
        input_out = os.path.join(out, f'{src_space}_input/{target_space}_to_{src_space}_input/images/')
        if not os.path.exists(input_out):
            os.makedirs(input_out)
        for i in range(AphiI_to_input.shape[1]):
            AphiI_to_input_ = AphiI_to_input[:, i, None, ...]
            xI_ = [torch.tensor([xI[0][i], xI[0][i]+10]), xI[1], xI[2]]
            emlddmm.write_vtk_data(os.path.join(input_out, f'{target_space}_{target_img}_to_{src_space}_input_{slice_names[i]}.vtk'), xI_, AphiI_to_input_, f'{target_space}_{target_img}_to_{src_space}_input_{slice_names[i]}')
        
    else:
        if J_title == 'slice_dataset':
            img_out = os.path.join(out, f'{src_space}/{target_space}_input_to_{src_space}/images/')
            if not os.path.exists(img_out):
                os.makedirs(img_out)

        else:
            img_out = os.path.join(out, f'{src_space}/{target_space}_to_{src_space}/images/')
            if not os.path.exists(img_out):
                os.makedirs(img_out)
        emlddmm.write_vtk_data(os.path.join(img_out, f'{target_space}_{target_img}_to_{src_space}.vtk'), x, AphiI, f'{target_space}_{target_img}_to_{src_space}')

    # save text file of transformation order
    if I_title == "slice_dataset":
        # save transformation sequence to input directory
        with open(os.path.join(out, f'{src_space}_input/{target_space}_to_{src_space}_input/{target_space}_to_{src_space}_transform_seq.txt'), 'w') as f:
            for transform in reversed(transformation_seq[1:]):
                f.write(str(transform) + ', ')
            f.write(str(transformation_seq[0]))
        # save transformation sequence to RECONSTRUCTED directory
            with open(os.path.join(out, f'{src_space}_registered/{target_space}_to_{src_space}_registered/{target_space}_to_{src_space}_transform_seq.txt'), 'w') as f:
                for transform in reversed(transformation_seq[1:]):
                    f.write(str(transform) + ', ')
                f.write(str(transformation_seq[0]))

    else:
        if J_title == 'slice_dataset':
            output_name = os.path.join(out, f'{src_space}/{target_space}_registered_to_{src_space}/{target_space}_registered_{target_img}_to_{src_space}_transform_seq.txt')
        else:
            output_name = os.path.join(out, f'{src_space}/{target_space}_to_{src_space}/{target_space}_{target_img}_to_{src_space}_transform_seq.txt')
        with open(output_name, 'w') as f:
            for transform in reversed(transformation_seq[1:]):
                f.write(str(transform) + ', ')
            f.write(str(transformation_seq[0]))

    return


def run_registrations(reg_list):
    """ Run Registrations

    Runs a sequence of registrations given by reg_list. Saves transforms, qc images, reconstructed images,
    displacement fields, and determinant of Jacobian of displacements. Also builds and writes out the transform graph. 

    Parameters
    ----------
    reg_list : list of dicts
        each dict in reg_list specifies the source image path, target image path,
        source and target space names, registration configuration settings, and output directory.
    
    Returns
    -------
    reg_graph : emlddmm graph

    Example
    -------
    >>> reg_list = [{'registration':[['CCF','average_template_50'],['MRI','masked']],
                     'source': '/path/to/average_template_50.vtk',
                     'target': '/path/to/HR_NIHxCSHL_50um_14T_M1_masked.vtk',
                     'config': '/path/to/configMD816_MR_to_CCF.json',
                     'output': 'outputs/example_output'},
                    {'registration':[['MRI','masked'], ['HIST','Nissl']],
                     'source': '/path/to/HR_NIHxCSHL_50um_14T_M1_masked.vtk',
                     'target': '/path/to/MD816_STIF',
                     'config': '/path/to/configMD816_Nissl_to_MR.json',
                     'output': 'outputs/example_output'}]
    >>> run_registrations(reg_list)

    """

    # initialize graph

    graph = Graph()
    for i in reg_list:
            for j in [i['registration'][0][0], i['registration'][1][0]]: # for src and target space names in each registration
                if j not in graph.spaces: 
                    graph.add_space(j)
    graph.adj = [{} for i in range(len(graph.spaces))]

    # perform registrations
    for r in reg_list:
        source = r['source']
        target = r['target']
        registration = r['registration']
        config = r['config']
        output_dir = r['output']
        print(f"registering {source} to {target}")
        with open(config) as f:
            config = json.load(f)
        I = emlddmm.Image(space=registration[0][0], name=registration[0][1], fpath=source)
        if I.title == 'slice_dataset': # if series to series, both images must share the same coordinate grid
            J = emlddmm.Image(space=registration[1][0], name=registration[1][1], fpath=target, mask=True, x=I.x)
        else:
            J = emlddmm.Image(space=registration[1][0], name=registration[1][1], fpath=target, mask=True)
        # add domains to graph before downsampling for later
        graph.spaces[I.space][1] = I.x
        graph.spaces[J.space][1] = J.x
        # initial downsampling
        downIs = config['downI']
        downJs = config['downJ']
        mindownI = np.min(np.array(downIs),0)
        mindownJ = np.min(np.array(downJs),0)
        I.x, I.data, I.mask = I.downsample(mindownI)
        J.x, J.data, J.mask = J.downsample(mindownJ)
        downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
        downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
        # update our config variable
        config['downI'] = downIs
        config['downJ'] = downJs
        # registration
        output = emlddmm.emlddmm_multiscale(I=I.data,xI=[I.x],J=J.data,xJ=[J.x],W0=J.mask,full_outputs=False,**config)
        '''
        for series to series:
            1) Save rigid transforms in {source}_input/{target}_to_{source}_input
            2) Save  qc source input and target to source_input in {source}_input/{target}_to_{source}_input/qc/
            3) Save qc target input and source to target_input in {target}_input/{source}_to_{target}_input/qc/

        for volume (source) to series (target):
            1) Save rigid transforms (R) in {target}_registered/{target}_input_to_registered/transforms/
            2) A.txt and velocity.vtk map points in source to match target, which are
            used to resmaple target images in source space. Save them in {source}/{target}_registered_to_{source}/transforms/
            3) Save qc source original and target to source in {source}/{target}_registered_to_{source}/qc/
            4) Save qc target_input and source to target_input in {target}_input/{source}_to_{target}_input/qc/
            5) Save qc target_registered and source to target_registered in {target}_registered/{source}_to_{target}_registered/qc/

        for volume to volume:
            1) Save A.txt and velocity.vtk in {source}/{target}_to_{source}/transforms/
        '''
        emlddmm.write_transform_outputs(output_dir, output[-1], I, J)
        emlddmm.write_qc_outputs(output_dir, output[-1], I, J)

        A = emlddmm.Transform(output[-1]['A'], 'f')
        Ai = emlddmm.Transform(output[-1]['A'], 'b')
        xv = output[-1]['xv']
        v = output[-1]['v']
        phi = emlddmm.Transform(v, direction='f', domain=xv)
        phii = emlddmm.Transform(v, direction='b', domain=xv)
        if 'A2d' in output[-1]:
            A2d = emlddmm.Transform(output[-1]['A2d'], 'f')
            A2di = emlddmm.Transform(output[-1]['A2d'], 'b')
            graph.add_edge([phi, A, A2d], I.space, J.space)
            graph.add_edge([A2di, Ai, phii], J.space, I.space)
        else:    
            graph.add_edge([phi, A], I.space, J.space)
            graph.add_edge([Ai, phii], J.space, I.space)

    return graph


def main():
    """ Main

    Main function for parsing input arguments, calculating registrations and applying transformations.

    Example
    -------
    $ python transformation_graph.py --infile GDMInput.json

    """

    help_string = "Arg parser looks for one argument, \'--infile\', which is a JSON file with the following entries: \n\
1) \"space_image_path\": a list of lists, each containing the space name, image name, and path to an image or image series. (Required)\n\
2) \"registrations\": a list of lists, each containing two space-image pairs to be registered. e.g. [[[\"HIST\", \"nissl\"], [\"MRI\", \"masked\"]],\n\
                                                                                                    [[\"MRI\", \"masked\"], [\"CCF\", \"average_template_50\"]],\n\
                                                                                                    [[\"MRI\", \"masked\"], [\"CT\", \"masked\"]]]\n\
                    If registrations were previously computed, this argument may be omitted, and the \"graph\" argument can be used to perform additional reconstructions.\n\
3) \"configs\": list of paths to registration config JSON files, the order of which corresponds to the order of registrations listed in the previous value. (Required if computing registrations)\n\
4) \"output\": output directory which will be the output hierarchy root. If none is given, it is set to the current directory.\n\
5) \"transforms\": transforms to apply after they are computed from registration. Only necessary if \"transform_all\" is False. Format is the same as \"registrations\".\n\
6) \"transform_all\": bool. Performs all possible reconstructions given the transform graph.\n\
7) \"graph\": path to \"graph.p\" pickle file saved to output after performing registrations. (Only required for reconstructing images from previously computed registrations)"

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--infile', nargs=1,
                        help=help_string,
                        type=argparse.FileType('r'))
    arguments = parser.parse_args()
    input_dict = json.load(arguments.infile[0])
    output = input_dict["output"] if "output" in input_dict else os.getcwd()
    if not os.path.exists(output):
        os.makedirs(output)
    # save out the json file used for input
    with open(os.path.join(output, "infile.json"), "w") as f:
        json.dump(input_dict, f, indent="")

    # get space_image_path
    try:
        space_image_path = input_dict["space_image_path"]
    except KeyError:
        print("space_image_path is a required argument. It is a list of images,\
             with each image being a list of the format: [\"space name\", \"image name\", \"image path\"]")
    # convert space_image_path to dictionary of dictionaries. (image_name-path key-values in a dict of space-img key-values)
    sip = {} # space-image-path dictionary
    for i in range(len(space_image_path)):
        if not space_image_path[i][0] in sip:
            sip[space_image_path[i][0]] = {}
        new_img = {space_image_path[i][1]: space_image_path[i][2]}
        sip[space_image_path[i][0]].update(new_img)

    # compute registrations
    if "registrations" in input_dict:
        try:
            configs = input_dict["configs"] # if registrations then there must also be configs
        except KeyError:
            ("configs must be included in input with registrations. configs is a list of full paths to JSON registration configuration files.")
        registrations = input_dict["registrations"]
        reg_list = [] # a list of dicts specifying inputs for each registration to perform
        for i in range(len(registrations)):
            src_space = registrations[i][0][0]
            src_img = registrations[i][0][1]
            src_path = sip[src_space][src_img]
            target_space = registrations[i][1][0]
            target_img = registrations[i][1][1]
            target_path = sip[target_space][target_img]

            reg_list.append({'registration': registrations[i], # registration format [[src_space, src_img], [target_space, target_img]]
                            'source': src_path,
                            'target': target_path,
                            'config': configs[i],
                            'output': output})
        print('registration list: ', reg_list, '\n')
        print('running registrations...')
        graph = run_registrations(reg_list)
        # save graph. Note: if a graph was supplied as an argument, it will be merged with the new one before saving.
        if "graph" in input_dict:
            tmp_graph = pickle.load(input_dict["graph"])
            # TODO: merge tmp_graph with graph
        with open(os.path.join(output, 'graph.p'), 'wb') as f:
            pickle.dump(graph, f)

    if "transform_all" in input_dict and input_dict["transform_all"] == True:
        for i in range(len(reg_list)):
            I = emlddmm.Image(reg_list[i]['registration'][0][0], reg_list[i]['registration'][0][1], reg_list[i]['source'])
            J = emlddmm.Image(reg_list[i]['registration'][1][0], reg_list[i]['registration'][1][1], reg_list[i]['target'])
            graph_reconstruct(graph, output, I, J)

    if "transforms" in input_dict:
        transforms = input_dict["transforms"]
        # this requires a graph which can be output from run_registrations or included in input json
        if "graph" in input_dict:
            graph = pickle.load(input_dict["graph"])
        assert "graph" in locals(), "\"graph\" argument is required when only applying new transforms."
        for t in transforms:
            spaceA = t[i][0][0]
            img_name = t[i][0][1]
            spaceB = t[i][1][0]
            img_path = sip[spaceA][img_name]
            image =  Image(spaceA, img_name, img_path)

            # TODO: write out reconstruction

    # for each registration save out the transforms (A, v), and qc images

    return

if __name__ == "__main__":
    main()

'''

 1) load images (I = image("space", "name", fpath))
 2) normalize (I.normalize(norm="mean"))
 3) downsample (I.downsample(down))
 4) load configuration parameters
 5) register images
 6) write out transforms & construct graph
 7) write out graph
 8) Reconstruct images by composing transforms according to shortest paths on the graph
 9) Save images and jacobians of composed transforms

'''



