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
    spaces : dict
        Integer keys map to the space name and the domain of the corresponding image space.
    adj: list
       Adjacency list. List of dictionaries holding the transforms needed to map between connecting spaces.
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
            print("Given target and source are not connected") # TODO: make this more informative
            
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


    def merge(self, new_graph):
        ''' Merge two graphs

        Parameters
        ----------
        new_graph : emlddmm Graph object

        Returns
        -------
        graph : emlddmm Graph object
            Current graph merged with the new graph.
        '''
        graph = self
        # merge spaces dict
        id_map = {} # we need a dict to map new space indices to the existing ones
        for key, value in new_graph.spaces.items():
            if key in graph.spaces:
                id_map.update({value[0]:graph.spaces[key][0]})
            else:
                id_map.update({value[0]:len(graph.spaces)})
                value[0] = len(graph.spaces)
                graph.spaces.update({key:value})
                graph.adj.append({}) # add a node in the adjacency list for each new space

        # merge adjacency list
        for i, node in enumerate(new_graph.adj): # for each node in the graph
            src = id_map[i] # get the original index for the space
            for j in node: # and for each node to which it connects
                target = id_map[j]
                transform = node[j]
                graph.adj[src].update({target:transform}) # update the graph

        return graph


def graph_reconstruct(graph, out, I, target_space, target_fnames=[]):
    ''' Apply Transformation

    Applies affine matrix and velocity field transforms to map source points to target points. Saves displacement field from source points to target points
    (i.e. difference between transformed coordinates and input coordinates), and determinant of Jacobian for 3d source spaces. Also saves transformed image in vtk format.

    Parameters
    ----------
    graph : emlddmm Graph object
    out: str
        path to registration outputs parent directory
    I : emlddmm Image   
    target_space : str
        name of the space to which image I will be transformed.
    target_fnames : list
        list of file names; only necessary if target is a series of 2d slices.

    '''
    jacobian = lambda X,dv : np.stack(np.gradient(X, dv[0],dv[1],dv[2], axis=(1,2,3))).transpose(2,3,4,0,1)

    dtype = torch.float
    device = 'cpu'
    # convert data to torch
    # J.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in J.x]
    # J.data = torch.as_tensor(J.data, dtype=dtype, device=device)
    xJ = graph.spaces[target_space][1]
    xJ = [torch.as_tensor(x, dtype=dtype, device=device) for x in xJ]
    target_space_idx = graph.spaces[target_space][0]
    I.x = [torch.as_tensor(x, dtype=dtype, device=device) for x in I.x]
    I.data = torch.as_tensor(I.data, dtype=dtype, device=device)
    src_space_idx = graph.spaces[I.space][0]

    # backward transform
    path = graph.shortest_path(target_space_idx, src_space_idx)
    transforms = graph.transforms(path)
    XJ = torch.stack(torch.meshgrid(xJ, indexing='ij'))
    fXJ = graph.map_points(target_space, transforms)
    fI = graph.map_image(target_space, I.space, I.data, transforms)
    '''
    Three cases:

    1) series to series
        Save reconstructions of I slices in J space. 
        Apply A2di to J points and resample I.
    2) volume to series
        A) Save volume to registered images in {target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/images/,
        and volume to input images in {target_space}_input/{I.space}_{I.name}_to_{target_space}_input/images/
        B) Save volume to registered and volume to input displacement in {target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/transforms/ and 
        {target_space}_input/{I.space}_{I.name}_to_{target_space}_input/transforms/, respectively.
    3) series to volume
        A) save series input to registered space images in {I.space}_registered/{I.space}_{I.name}_input_to_{I.space}_registered/images/
        C) Save series to volume image in {target_space}/{I.space}_{I.name}_input_to_{I.space}/images/
        D) Save series to volume detjac and displacement in {target_space}/{I.space}_{I.name}_registered_to_{target_space}/transforms/
    4) volume to volume 
        A) Save out I to J image in {target_space}/{I.space}_to_{target_space}/images.
        B) Save out I to J detjac and displacement in {target_space}/{I.space}_to_{target_space}/transforms/
    '''
    from_series = I.title == 'slice_dataset'
    to_series = transforms[0].data.ndim == 3 # we don't have the image title so we need to look at the transforms
    if from_series and to_series:
        # series to series
        # Assumes J and I have the same space dimensions
        I_to_J_out = os.path.join(out, f'{target_space}_input/{I.space}_input_to_{target_space}_input/images/')
        if not os.path.exists(I_to_J_out):
            os.makedirs(I_to_J_out)
        for i in range(I.data.shape[1]):
            x = [[I.x[0][i], I.x[0][i]+10], I.x[1], I.x[2]]
            # I to J
            img = fI[:, i, None, ...]
            title = f'{I.space}_input_{I.fnames()[i]}_to_{target_space}_input_{target_fnames[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_J_out, f'{I.space}_input_{I.fnames()[i]}_to_{target_space}_input_{target_fnames[i]}.vtk'), x, img, title)

    elif to_series:
        # volume to series
        # we need I transformed to J registered space
        path = graph.shortest_path(graph.spaces[target_space][0], graph.spaces[I.space][0])
        # omit the first 2d series transforms (R^-1) which takes points from 2d to 2d or input to registered.
        # in this case we can just remove all 2d series transforms.
        transforms = graph.transforms(path)
        for i, t in enumerate(transforms):
            if t.data.ndim == 3:
                del transforms[i]
        phiiAiXJ = graph.map_points(target_space, transforms)
        AphiI = graph.map_image(target_space, I.space, I.data, transforms)
        # get I to J registered and I to J input displacements
        reg_disp = (phiiAiXJ - XJ)[None]
        input_disp = (fXJ - XJ)[None]

        # setup output paths
        I_to_Ji_out = os.path.join(out, f'{target_space}_input/{I.space}_{I.name}_to_{target_space}_input/images/')
        if not os.path.exists(I_to_Ji_out):
            os.makedirs(I_to_Ji_out)
        I_to_Jr_out = os.path.join(out, f'{target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/images/')
        if not os.path.exists(I_to_Jr_out):
            os.makedirs(I_to_Jr_out)
        reg_disp_out = os.path.join(out, f'{target_space}_registered/{I.space}_{I.name}_to_{target_space}_registered/transforms/')
        if not os.path.exists(reg_disp_out):
            os.makedirs(reg_disp_out)
        input_disp_out = os.path.join(out, f'{target_space}_input/{I.space}_{I.name}_to_{target_space}_input/transforms/')
        if not os.path.exists(input_disp_out):
            os.makedirs(input_disp_out)
        for i in range(len(xJ[0])):
            x = [[xJ[0][i], xJ[0][i]+10], xJ[1], xJ[2]]
            # volume to input series
            # save image
            img = fI[:, i, None, ...]
            title = f'{I.space}_{I.name}_to_{target_space}_input_{target_fnames[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_Ji_out, title + '.vtk'), x, img, title)
            # save displacement
            disp = input_disp[:, :, i, None]
            title = f'{target_space}_input_{target_fnames[i]}_to_{I.space}_displacement'
            emlddmm.write_vtk_data(os.path.join(input_disp_out, title + '.vtk'), x, disp, title)
            # volume to registered series
            img = AphiI[:, i, None, ...]
            title = f'{I.space}_{I.name}_to_{target_space}_registered_{target_fnames[i]}'
            emlddmm.write_vtk_data(os.path.join(I_to_Jr_out, title + '.vtk'), x, img, title)
            # save displacement
            disp = reg_disp[:, :, i, None]
            title = f'{target_space}_registered_{target_fnames[i]}_to_{I.space}_displacement'
            emlddmm.write_vtk_data(os.path.join(reg_disp_out, title + '.vtk'), x, disp, title)

    
    elif from_series:
        # series to volume
        # I input to I registered space
        path = graph.shortest_path(graph.spaces[target_space][0], graph.spaces[I.space][0])
        # get the last 2d series transforms (R) which take points from 2d to 2d or registered to input
        transforms = graph.transforms(path)
        idx = 0
        for i,t in enumerate(transforms[::-1]):
            if t.data.ndim != 3:
                idx = i
                break
        transforms = transforms[-idx:]
        XI = torch.stack(torch.meshgrid(I.x, indexing='ij'))
        RXI = emlddmm.compose_sequence(transforms, XI)
        RiI = emlddmm.interp(I.x, I.data, RXI)

        Ii_to_Ir_out = os.path.join(out, f'{I.space}_registered/{I.space}_input_to_{I.space}_registered/images/')
        if not os.path.exists(Ii_to_Ir_out):
            os.makedirs(Ii_to_Ir_out)

        # input to registered images
        img = RiI[:, i, None, ...]
        title = f'{I.space}_input_{I.fnames()[i]}_to_{I.space}_registered_{I.fnames()[i]}'
        emlddmm.write_vtk_data(os.path.join(Ii_to_Ir_out, title + '.vtk'), I.x, img, title)

        # I to J
        img = fI
        title = f'{I.space}_{I.name}_input_to_{target_space}'
        I_to_J_imgs = os.path.join(out, f'{target_space}/{I.space}_{I.name}_input_to_{target_space}/images/')
        if not os.path.exists(I_to_J_imgs):
            os.makedirs(I_to_J_imgs)
        emlddmm.write_vtk_data(os.path.join(I_to_J_imgs, title + '.vtk'), xJ, img, title)
        # disp
        # we need J to I registered points
        disp = (fXJ - XJ)[None]
        title = f'{I.space}_{I.name}_registered_to_{target_space}_displacement'
        I_to_J_transforms = os.path.join(out, f'{target_space}/{I.space}_{I.name}_registered_to_{target_space}/transforms/')
        if not os.path.exists(I_to_J_transforms):
            os.makedirs(I_to_J_transforms)
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, disp, title)
        # determinant of jacobian
        dv = [(x[1]-x[0]) for x in xJ]
        jac = jacobian(fXJ, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{I.space}_{I.name}_registered_to_{target_space}_detjac'
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, detjac, title)
    else:
        # volume to volume
        # I to J
        # save image
        img = fI
        title = f'{I.space}_{I.name}_to_{target_space}'
        I_to_J_imgs = os.path.join(out, f'{target_space}/{I.space}_{I.name}_to_{target_space}/images/')
        if not os.path.exists(I_to_J_imgs):
            os.makedirs(I_to_J_imgs)
        emlddmm.write_vtk_data(os.path.join(I_to_J_imgs, title + '.vtk'), xJ, img, title)
        # save displacement
        disp = (fXJ - XJ)[None]
        title = f'{I.space}_{I.name}_to_{target_space}_displacement'
        I_to_J_transforms = os.path.join(out, f'{target_space}/{I.space}_{I.name}_to_{target_space}/transforms/')
        if not os.path.exists(I_to_J_transforms):
            os.makedirs(I_to_J_transforms)
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, disp, title)
        # save determinant of jacobian
        dv = [(x[1]-x[0]) for x in xJ]
        jac = jacobian(fXJ, dv)
        detjac = np.linalg.det(jac)[None]
        title = f'{I.space}_{I.name}_to_{target_space}_detjac'
        emlddmm.write_vtk_data(os.path.join(I_to_J_transforms, title + '.vtk'), xJ, detjac, title)


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
    # convert space_image_path to dictionary of dictionaries. (image_name:path key-values in a dict of space:img key-values)
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
            with open(input_dict["graph"], 'rb') as f:
                tmp_graph = pickle.load(f)
            graph  = graph.merge(tmp_graph)
        with open(os.path.join(output, 'graph.p'), 'wb') as f:
            pickle.dump(graph, f)

    if "transform_all" in input_dict and input_dict["transform_all"] == True:
        for src_space in sip:
            for src_image in sip[src_space]:
                src_path = sip[src_space][src_image]
                I = emlddmm.Image(src_space, src_image, src_path, x=graph.spaces[src_space][1])
                # reconstruct in every other space
                for target_space in [n for n in sip if n != src_space]:
                    target_image = list(sip[target_space].keys())[0] # get the first image. This is just to get file names if it is a series.
                    target_path = sip[target_space][target_image]
                    if os.path.splitext(target_path)[-1] == '':
                        fnames = emlddmm.fnames(target_path)
                        graph_reconstruct(graph, output, I, target_space, target_fnames=fnames)
                    else:
                        graph_reconstruct(graph, output, I, target_space)

    elif "transforms" in input_dict:
        transforms = input_dict["transforms"]
        # this requires a graph which can be output from run_registrations or included in input json
        if "graph" in input_dict:
            with open(input_dict["graph"], 'rb') as f:
                graph = pickle.load(f)
        assert "graph" in locals(), "\"graph\" argument is required when only applying new transforms."
        for t in transforms:
            I = emlddmm.Image(t[0][0], t[0][1], sip[t[0][0]][t[0][1]], x=graph.spaces[t[0][0]][1])
            target_space = t[1][0]
            target_image = t[1][1]
            target_path = sip[target_space][target_image]
            if os.path.splitext(target_path)[-1] == '':
                fnames = emlddmm.fnames(target_path)
                graph_reconstruct(graph, output, I, target_space, target_fnames=fnames)
            else:
                graph_reconstruct(graph, output, I, target_space)

    return

if __name__ == "__main__":
    main()



