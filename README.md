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

#### Dataset lists



## Config files

In python, configuration options are passed as dictionaries.  For command line use these are stored in json files.  See the examples for examples of various parameters.

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

### Reconstruction tuples



## Output data format





## Python Interface

## Comand Line Interface

## Pipelines

Link here to several pipelines.

Mouse:

3D registration pipeline with STP

3D to 2D registration pipeline with Nissl series

3D to 2D registration pipeline with alternating series


Marmoset:

3D registration MR to MR



