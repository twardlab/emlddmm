# EMLDDMM
## Introduction
Expectation Maximization Large Deformation Diffeomorphic Metric Mapping is an image registration method for aligning datasets in the presence of differing contrast profiles and missing tissue or artifacts.

It uses an expectation maximization algorithm (EM) [ref] to handle missing data, and leverages the powerful Large Deformation Diffeomorphic Metric Mapping (LDDMM) paradaigm [ref] to ensure mappings are diffeomorphisms [ref], and are generated in a Reimannian framework suitable for statistical analysis [ref].

These concepts were brought together into the EMLDDMM algorithm described in  [ref,ref].

Our package is designed for 3D to 3D image registration, or 3D to 2D serial sections.

## File formats

### 3D data
We use vtk data as a standard, and use ([vtk simple legacy format](https://kitware.github.io/vtk-examples/site/VTKFileFormats/) because it has a human readable header.  It supports images and vector fields, as well as polydata (points, edges, triangulations) under a common standard.  For visualization we use [paraview](https://www.paraview.org/), or [itksnap](http://www.itksnap.org/pmwiki/pmwiki.php).

For input data we include nibabel and pynrrd as a dependency, and support several other formats supported by these libraries.

### 2D serial section data
For 2D serial section datasets images should be stored in standard imaging formats (i.e. to be read by matplotlib's imread function).  While our pipelines do support downsampling to desired resolutions, these serial section images are expected.

#### JSON Sidecar files
2D images have their geometry specified in a json sidecar file. Such sidecar files are inspired by the BIDS (ref) standard, and contain information typically in an NRRD header (ref),

Note that each 2D image is modeled as a 3D image with a single slice.  An example is shown here.
```
{
  "DataFile": "MD787_small_nissl/MD787-N27-2019.03.28-22.55.54_MD787_2_0080.png",
  "Type": "Float32",
  "Dimension": 3,
  "Endian": "big",
  "Sizes": [
    3,
    392,
    480,
    1
  ],
  "Space": "inferior-right-posterior",
  "SpaceDimension": 3,
  "SpaceUnits": [
    "um",
    "um",
    "um"
  ],
  "SpaceDirections": [
    "none",
    [
      44.160000000000004,
      0.0,
      0.0
    ],
    [
      0.0,
      44.160000000000004,
      0.0
    ],
    [
      0.0,
      0.0,
      200
    ]
  ],
  "SliceThickness": 10.0,
  "SpaceOrigin": [
    -8633.28,
    -10576.32,
    -120100.0
  ]
}
```
#### Dataset lists
Since sections may be missing or require other comments, we include a tsv file describing every slice in the dataset.

```
sample_id 	 participant_id 	 species 	 status
MD787-N7-2019.03.28-22.05.43_MD787_2_0020.png	MD787	Mus Musculus	present
MD787-N14-2019.03.28-22.20.46_MD787_1_0040.png	MD787	Mus Musculus	present
MD787-N20-2019.03.28-22.36.39_MD787_3_0060.png	MD787	Mus Musculus	present
MD787-N27-2019.03.28-22.55.54_MD787_2_0080.png	MD787	Mus Musculus	present
MD787-N34-2019.03.28-23.15.58_MD787_1_0100.png	MD787	Mus Musculus	present
MD787-N40-2019.03.28-23.33.43_MD787_3_0120.png	MD787	Mus Musculus	present
MD787-N47-2019.03.28-23.54.40_MD787_2_0140.png	MD787	Mus Musculus	present
MD787-N54-2019.03.29-00.15.46_MD787_1_0160.png	MD787	Mus Musculus	present
MD787-N60-2019.03.29-00.33.42_MD787_3_0180.png	MD787	Mus Musculus	present
MD787-N67-2019.03.29-00.56.05_MD787_2_0200.png	MD787	Mus Musculus	present
MD787-N74-2019.03.29-01.18.34_MD787_1_0220.png	MD787	Mus Musculus	present
MD787-N80-2019.03.29-01.36.50_MD787_3_0240.png	MD787	Mus Musculus	present
MD787-N87-2019.03.29-01.57.37_MD787_2_0260.png	MD787	Mus Musculus	present
MD787-N94-2019.03.29-02.19.41_MD787_1_0280.png	MD787	Mus Musculus	present
MD787-N100-2019.03.29-02.40.34_MD787_3_0300.png	MD787	Mus Musculus	present
MD787-N107-2019.03.29-03.04.17_MD787_2_0320.png	MD787	Mus Musculus	present
MD787-N114-2019.03.29-03.28.07_MD787_1_0340.png	MD787	Mus Musculus	present
MD787-N120-2019.03.29-03.49.00_MD787_3_0360.png	MD787	Mus Musculus	present
```


## Config files

In python, configuration options are passed as dictionaries.  For command line use these are stored in json files.  See the examples for examples of various parameters.

Here is an example with typical parameters.  When arguments are separated by commas, they refer to each iteration of a multi scale (coarse to fine) approach.

```
{
    "n_iter":[1000,200],
    "downI":[[4,4,4],[2,2,2]],
    "downJ":[[4,4,4],[2,2,2]],        
    "a":[200.0],
    "sigmaR":[5e6],
    "sigmaM":[2.0],
    "sigmaB":[4.0],
    "sigmaA":[6.0],
    "ev":[1e-0],
    "eA":[1e6],
    "priors":[[0.9,0.05,0.05]],
    "update_muA":[0],
    "update_muB":[0],
    "muB":[0.0],
    "update_sigmaM":[0],
    "update_sigmaA":[0],
    "update_sigmaB":[0],
    "order":[3],
    "n_draw":[50],
    "n_e_step":[3],    
    "v_start":[500,0]
}

```
## Spaces


### 3D spaces

#### Mouse

#### Marmoset
TODO

#### Human
TODO

### 2D spaces
#### Input histology space


#### Registered histology space
In our mapping pipeline we apply a rigid transform to each histology slice, so that each slice aligns with its neighors, and the overall geometry matches the 3D dataset being registered.

Since no data was acquired in this space, it can be defined by various conventions.



## Input arguments
We support pipelines for registering several datasets to each other, and reconstructing data from one dataset in the space of any other dataset.
### Names of spaces
Registrations are computed between pairs of spaces.  Each space should be given a unique name. (e.g. "atlas", "CT", "exvivoMRI","invivoMRI", "Histology").

### Names of images
Each space may have more than one imaging dataset sampled in it (for example multiple MRI scans with different contrasts).  Each image within a space should be given a unique name.  (e.g. "exvivoMRI -> T1", "exvivoMRI -> T2", "invivoMRI -> T1", "Histology")

### Registration tuples
To register a complex multimodal datasets, we specify a list of (space/image to map from, space/image to map to ) tuples. These correspond to edges in a graph and should span the set of spaces.  This set of transformations will be computed using an optimization procedure.

### Reconstruction tuples
After transforamtions are computed, we can reconstruct data from one space in any other space. Tuples of the form (space/image to map from, space to map to) are specified. Given the registration tuples, a path of transformations will be computed, which may involve the composition of more than one calculated transform.


## Output data format

Directory 



## Python Interface
Example TODO

## Comand Line Interface
Example TODO

## Pipelines

Link here to several pipelines. TODO

Mouse:

3D registration pipeline with STP

3D to 2D registration pipeline with Nissl series

3D to 2D registration pipeline with alternating series


Marmoset:

3D registration MR to MR
TODO

Human:
TODO


