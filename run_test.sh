#! /bin/bash

ATLAS=/home/dtward/data/AllenInstitute/allen_vtk/ara_nissl_50_bregma.vtk
LABEL=/home/dtward/data/AllenInstitute/allen_vtk/annotation_50_bregma.vtk
TARGET=../microscopy_v2/microscopy/
CONFIG=config787.json
OUT=test_out
MODE=register


python emlddmm.py -m $MODE -c $CONFIG -o $OUT -a $ATLAS -l $LABEL -t $TARGET 
