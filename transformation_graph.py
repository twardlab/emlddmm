import emlddmm
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt

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
  
  
# utility function to print the shortest distance
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


def reg_3d_3d(source, target, config, out):
    atlas_name = source
    target_name = target
    config_file = config
    output_dir = out
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

    # initial downsampling so there isn't so much on the gpu
    mindownI = np.min(np.array(downIs),0)
    xI,I = emlddmm.downsample_image_domain(xI,I,mindownI)
    downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
    dI = [x[1]-x[0] for x in xI]
    nI = np.array(I.shape,dtype=int)
    # update our config variable
    config['downI'] = downIs

    # target
    xJ,J,title,names = emlddmm.read_data(target_name)
    J = J.astype(float)
    J /= np.mean(np.abs(J))
    xJ = [x for x in xJ]
    dJ = np.array([x[1]-x[0] for x in xJ])
    J = J.astype(float)#**0.25
    W0 = np.ones_like(J[0])

    # initial downsampling so there isn't so much on the gpu
    mindownJ = np.min(np.array(downJs),0)
    xJ,J = emlddmm.downsample_image_domain(xJ,J,mindownJ)
    W0 = emlddmm.downsample(W0,mindownJ)
    downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
    dJ = [x[1]-x[0] for x in xJ]
    nJ = np.array(J.shape,dtype=int)
    # update our config variable
    config['downJ'] = downJs

    device = 'cuda:0'
    #device = 'cpu'
    output = emlddmm.emlddmm_multiscale(I=I,xI=[xI],J=J,xJ=[xJ],W0=W0,device=device,**config)
    #write outputs
    emlddmm.write_transform_outputs(output_dir,output[-1])
    emlddmm.write_qc_outputs(output_dir,output[-1],xI,I,xJ,J)

    return

def reg_3d_2d(source, target, config, out):
    
    print()

    return


def run_registrations(reg_list):
    # input: list of dicts of sources, targets, labels, configs, and the output dir 
    # Return: adj list, dict of space name keys and node number values

    # perform registration using reg_list as input
    for reg in reg_list:
        source = reg['source']
        target = reg['target']
        config = reg['config']
        out = reg['output_dir']
        reg_3d_3d(source, target, config, out)
         
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
        out = os.path.join(reg_list[i]['output_dir'],'transforms')
        add_edge(adj, spaces[src], spaces[dest], out)

    return adj, spaces


# Do new transform using composition of calculated transformations
def do_transformation(adj, spaces, src, dest, img=''):
    # input: image to be transformed, adjacency list and spaces dict from run_registration, source and destination space names
    # return: transfromed image

    path = findShortestPath(adj, spaces[src], spaces[dest], len(spaces))
    print("\nPath is:")

    for i in path:
        for key, value in spaces.items():
            if i == value:
                print(key, end=' ')

    transformation_seq = getTransformation(adj, path)
    print('\nTransformation sequence: ', transformation_seq)

    xI, I, title, names = emlddmm.read_data(img)
    Xin = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))
    Xout = emlddmm.compose_sequence([transformation_seq[0]], Xin)
    for transform in transformation_seq[1:]:
        Xout = emlddmm.compose_sequence([transform], Xout)
    AphiI = emlddmm.apply_transform_float(xI, I, Xout)

    return AphiI


# Driver program to test above functions
if __name__=='__main__':
     
    reg_list = [{'spacenames': ['MRI', 'ATLAS'], 'source': 'C:\\Users\\BGAdmin\\data\\MD816/HR_NIHxCSHL_50um_14T_M1_masked.vtk',\
                    'target': 'C:\\Users\\BGAdmin\\data\\Allen_Atlas_vtk/average_template_50.vtk', 'config': 'C:\\Users\\BGAdmin\\emlddmm\\configMD816_MR_to_CCF.json', 'output_dir': 'test_output1'}]

    src = 'MRI'
    dest = 'ATLAS'

    adj, spaces = run_registrations(reg_list)
    img = 'C:\\Users\\BGAdmin\\data\\MD816/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
    xI, _, _, _ = emlddmm.read_data(img)
    AphiI = do_transformation(adj=adj, spaces=spaces, src=src, dest=dest, img=img)

    # visualize transformation
    fig = emlddmm.draw(AphiI,xI)
    fig[0].suptitle('Transformed target')
    fig[0].canvas.draw()