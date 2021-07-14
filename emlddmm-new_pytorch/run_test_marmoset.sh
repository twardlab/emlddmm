#! /bin/bash

ATLAS=/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-mri-reorient.vtk
LABEL=/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-region_seg-reorient.vtk
TARGET=/home/dtward/data/csh_data/marmoset/m1229/M1229MRI/MRI/exvivo/HR_T2/HR_T2_CM1229F-reorient.vtk
CONFIG=config1229.json
OUT=test_out_marmoset
MODE=register


python emlddmm.py -m $MODE -c $CONFIG -o $OUT -a $ATLAS -l $LABEL -t $TARGET 
