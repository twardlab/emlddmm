# EMLDDMM
## Introduction
Expectation Maximization Large Deformation Diffeomorphic Metric Mapping is an image registration method for aligning datasets in the presence of differing contrast profiles and missing tissue or artifacts.

It uses an expectation maximization algorithm (EM) [ref] to handle missing data, and leverages the powerful Large Deformation Diffeomorphic Metric Mapping (LDDMM) paradaigm [ref] to ensure mappings are diffeomorphisms [ref], and are generated in a Reimannian framework suitable for statistical analysis [ref].

These concepts were brought together into the EMLDDMM algorithm described in  [ref,ref].


## Outline

## File formats

We use vtk data as a standard, and use legacy format because it has a human readable header (link).  It supports images and vector fields.

For input data we include nibabel and pynrrd as a dependency, and support any of these




## Config files


## Jupyter and command line interface

## Pipelines

Link here to several pipelines.

Mouse:

3D registration pipeline with STP

3D to 2D registration pipeline with Nissl series

3D to 2D registration pipeline with alternating series


Marmoset:

3D registration MR to MR


