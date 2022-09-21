from turtle import pd
import emlddmm
import numpy as np
import torch
import argparse
from argparse import RawTextHelpFormatter
import json
import os
import pickle
import re

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

    def __init__(self, nodes={}, edges=[]):
        self.nodes = nodes
        self.edges = edges
        pass

    def add_segment(self):
        """
        Add an edge to the graph.

        Parameters
        ----------

        Returns
        -------

        """
        pass

    def shortest(self, a, b):
        """
        Find the shortest path between two nodes.

        Parameters
        ----------
        spaceA : str
        spaceB : str

        Returns
        -------
        path : list
            list of nodes connecting spaceA to spaceB
        transforms : list
            list of inputs to compose_sequence, each being a list of two Transform objects
        """
        pass


class Image:
    def __init__(self, space, name, fpath, mask=None):
        self.space = space
        self.name = name
        self.x, self.data, self.title, self.names = emlddmm.read_data(fpath)
        self.data = self.data.astype(float)
        self.mask = mask
        if 'mask' in self.names:
            maskind = self.names.index('mask')
            self.mask = self.data[maskind]
            self.data = self.data[np.arange(self.data.shape[0])!=maskind]
        elif mask == True: # only initialize mask array if mask arg is True
            self.mask = np.ones_like(self.data[0])
        if self.title == 'slice_dataset':
            self.image_type = 'series'
        else:
            self.image_type = 'volume'
    
    def normalize(self, norm='mean'):
        if norm == 'mean':
            self.data /= np.mean(np.abs(self.data))
    
    def downsample(self, down):
        self.x, self.data = emlddmm.downsample_image_domain(self.x, self.data, down)
        if self.mask is not None:
            self.mask = emlddmm.downsample(self.mask,down)


def write_transform_outputs():
    pass


def write_qc_outputs():
    pass


def read_graph():
    pass


def write_graph():
    pass


def reconstruct(transforms, Xin, image):
    Xout = emlddmm.compose_sequence(transforms, Xin)
    I = emlddmm.apply_transform_float(image.x, image.data, Xout=Xout)
    # TODO: construct displacement and detjac
    disp = None
    detjac = None
    return I, disp, detjac

# TODO
def graph_reconstruct(graph, spaceA, spaceB, image):
    ''' Reconstruct image in desired space
    
    Parameters
    ----------

    Returns
    -------
    I : array
        image reconstructed in spaceB
    disp : array
        # TODO: does disp transform points from A to B or B to A?
    detjac : array
        determinant of Jacobian of the displacement field
    '''
    pass


def _fnames_from_dir(path):
    samples_tsv = os.path.join(path, "samples.tsv")
    fnames = []
    with open(samples_tsv,'rt') as f:
        for count,line in enumerate(f):
            line = line.strip()
            key = '\t' if '\t' in line else '    '
            if count == 0:
                continue
            fnames.append(os.path.splitext(re.split(key,line)[0])[0])
    return fnames


def run_registrations(reg_list):
    """ Run Registrations

    Runs a sequence of registrations given by reg_list. Saves transforms, qc images, reconstructed images,
    displacement fields, and determinant of Jacobian of displacements. Also builds and writes out the transform graph. 

    Parameters
    ----------
    reg_list : list of dicts
        each dict in reg_list specifies the source image path, target image path,
        source and target space names, registration configuration settings, and output directory.

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

    for r in reg_list:
        source = r['source']
        target = r['target']
        registration = r['registration']
        config = r['config']
        outdir = r['output']
        print(f"registering {source} to {target}")
        with open(config) as f:
            config = json.load(f)
        I = Image(space=registration[0][0], name=registration[0][1], fpath=source)
        J = Image(space=registration[1][0], name=registration[1][1], fpath=target, mask=True)
        I.normalize(norm='mean')
        J.normalize(norm='mean')
        # initial downsampling
        downIs = config['downI']
        downJs = config['downJ']
        mindownI = np.min(np.array(downIs),0)
        mindownJ = np.min(np.array(downJs),0)
        I.downsample(mindownI)
        J.downsample(mindownJ)
        downIs = [ list((np.array(d)/mindownI).astype(int)) for d in downIs]
        downJs = [ list((np.array(d)/mindownJ).astype(int)) for d in downJs]
        # update our config variable
        config['downI'] = downIs
        config['downJ'] = downJs
        # registration
        output = emlddmm.emlddmm_multiscale(I=I.data,xI=[I.x],J=J.data,xJ=[J.x],W0=J.mask,full_outputs=False,**config)
        # save transforms
        '''
        for volume to volume:
            1) Save A.txt and velocity.vtk in {source}/{target}_to_{source}

        for volume (source) to series (target):
            1) Save rigid transforms (R_i) in {target}_registered/{target}_input_to_registered.
            2) A.txt and velocity.vtk map points in source to match target, which are
            used to resmaple target images in source space. Save them in {source}/{target}_to_{source}
    
        for series to series:
            1) Save rigid transforms in {source}_input/{target}_to_{source}_input
        '''
        if I.image_type == 'series' and J.image_type == 'series':
            A2d_names = []
            source_fnames = _fnames_from_dir(source)
            target_fnames = _fnames_from_dir(target)
            for i in range(len(source_fnames)):
                print()
            emlddmm.write_transform_outputs(outdir, output, A2d_names=A2d_names)
            pass
        elif J.image_type == 'series':
            pass
        else:
            pass
        # TODO: Save displacement, determinant of Jacobian, and reconstructed images for each registration
        fI, disp, detjac = reconstruct(transforms, Xin, image)
        # build graph
        reg_graph = Graph()
        for i in range(len(reg_list)):
            reg_graph.add_segment() # TODO
        
        return reg_graph


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

    # load images
    images = []
    for i in space_image_path:
        images.append(Image(i[0], i[1], i[2]))
    # compute registrations
    if "registrations" in input_dict:
        registrations = input_dict["registrations"]
        reg_list = [] # a list of dicts specifying inputs for each registration to perform
        for i in range(len(registrations)):
            src_space = registrations[i][0][0]
            src_img = registrations[i][0][1]
            src_path = sip[src_space][src_img]
            target_space = registrations[i][1][0]
            target_img = registrations[i][1][1]
            target_path = sip[target_space][target_img]

            reg_list.append({'registration': registrations[i], # registrsation format [[src_space, src_img], [target_space, target_img]]
                            'source': src_path,
                            'target': target_path,
                            'config': configs[i],
                            'output': output})

        print('registration list: ', reg_list, '\n')
        try:
            configs = input_dict["configs"] # if registrations then there must also be configs
        except KeyError:
            ("configs must be included in input with registrations. configs is a list of full paths to JSON registration configuration files.")
        print('running registrations...')
        graph = run_registrations(reg_list)
        # save graph. Note: if a graph was supplied as an argument, it will be merged with the new one before saving.
        if "graph" in input_dict:
            tmp_graph = pickle.load(input_dict["graph"])
            # TODO: merge tmp_graph with graph
        with open(os.path.join(output, 'graph.p'), 'wb') as f:
            pickle.dump(graph, f)

    if "transform_all" in input_dict and input_dict["transform_all"] == True:
        print() # TODO
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
            fI, disp, detjac = graph_reconstruct(graph, spaceA, spaceB, image)
            # TODO: write out reconstruction

    # for each registration save out the transforms (A, v), and qc images

    return


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



