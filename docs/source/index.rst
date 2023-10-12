.. GDM documentation master file, created by
   sphinx-quickstart on Tue Apr 19 09:18:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GDM's documentation!
===============================

Generative diffeomorphic mapping (GDM) is a deformable image registration algorithm designed for aligning multimodal neuroimaging datasets to one another for subsequent analysis.  Our package has several important novel features including estimation of any differences in contrast or color between datasets, identification of missing tissues or artifactions, diverse geometries such as mapping 3D volumes to a sequence of 2D datasets, and complex multimodality registration setups described by transformation graphs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   introduction
   coordinate_systems
   installation
   documentation
   todo





Module and function documentation
=================================


Installing
==========
* requirements
* github
* pip

Examples
========
* Run emlddmm registration (jupyter notebook?)
* Run multi scale emlddmm registration (jupyter notebook?)
* Run multiple transformations (command line interface)
* Apply transformations to new datasets (command line interface)


Important Functions
===================

Python functions to be run interactively
----------------------------------------
* :py:func:`emlddmm.emlddmm`: Run the emlddmm algorithm, example here: TODO
* :py:func:`emlddmm.emlddmm_multiscale`: Run the emlddmm algorithm iteratively at different scales, example here: TODO

Command line functions
----------------------
* :py:mod:`transformation_graph_v01`: Run the transformation graph
