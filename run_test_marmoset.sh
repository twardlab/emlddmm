#! /bin/bash

ATLAS=/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-mri-reorient.vtk
LABEL=/home/dtward/data/csh_data/marmoset/Woodward_2018/bma-1-region_seg-reorient.vtk
TARGET=/home/dtward/data/csh_data/marmoset/m1229/M1229MRI/MRI/exvivo/HR_T2/HR_T2_CM1229F-reorient.vtk
CONFIG=config1229.json
OUT=test_out_marmoset
MODE=register
OUTPUTFORMAT='.nii'
NUM_THREADS=1

python emlddmm.py -m $MODE -c $CONFIG -o $OUT -a $ATLAS -l $LABEL -t $TARGET --output_image_format $OUTPUTFORMAT --num_threads $NUM_THREADS

# apply transforms
MODE=transform
OUT_=${OUT}/xformed
# this first line will transform atlas labels onto target image
python emlddmm.py -m $MODE -l $LABEL -t $TARGET -o $OUT_ -x $OUT -d b --output_image_format $OUTPUTFORMAT --num_threads $NUM_THREADS
# this second line will transform target into atlas space
python emlddmm.py -m $MODE -a $TARGET -t $ATLAS -o $OUT_ -x $OUT -d f --output_image_format $OUTPUTFORMAT --num_threads $NUM_THREADS
