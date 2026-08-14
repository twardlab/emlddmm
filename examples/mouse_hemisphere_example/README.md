= Overview =

This folder contains an example of how to register hemisphere brains from Yang Lab.

It contains 3 steps.  First we resample the data.  Second we register it.   Third we apply the results to create the outputs we want.

= Step 1: Downsampling =
Run the jupyter notebook `LoadData_20241017_5xFAD_Etv1_MORF3_MEF119_3_4m_RH`

This will register a stack of tifs in a directory specified by the varible `rootdir`.  On Daniel's system it looks like this:

    rootdir = '/media/dtward/YangLab_5TB_WDBlue/20241017_5xFAD_Etv1_MORF3_MEF119_3_4m_RH/9x/RES(15290x11566x3184)/459340/459340_505140'

Make sure that these files will be in the right order when sorted alphabetically.

We specify the resolution with the variable `res`.  This is used for setting pixel size, naming outputs, and display.

For the above file I used

    res = '9x'

Make sure the pixel size of your data is set properly.  The first number is the spacing between slices.

    dI0 = np.array([2.0, 1.8, 1.8]) # for 4x
    if res == '9x':
        dI0 = np.array([2.0,0.71,0.71])    




Set the desired output resolution with the variable `dIdes`

    dIdes = 25.0


The notebook will save downsampled data (the image, `Iddd`, and the pixel locations, `xI`) in the current working directory:

    savename = '_'.join([name,res,channel]) + '.npz'
    np.savez(savename,xI=np.array(xI,dtype=object),I=Iddd[None])



= Step 2: registration =

Run the notebook `register_march_6_2026`

Note arguments near the top about file names

    # load the target image
    target_filename = '20241017_5xFAD_Etv1_MORF3_MEF119_3_4m_RH_9x_C1.npz'
    outdir = '20241017_5xFAD_Etv1_MORF3_MEF119_3_4m_RH_9x_outputs_march_2026'
    atlas_name  = '/home/dtward/Documents/UCLA/dong/Nissl_10_symmetric.tif'
    outputname = 'yang_outputs_april_2026.npz'


And importantly, we need to set the initial guess for the affine transform in the cell labeled as such

    A0 = np.eye(4)
    A0[:3,:3] = emlddmm.orientation_to_orientation('PIR','LPS')
    A0[:3,-1] = [-2500,1000.0,0.0]

Use the provided figure to make sure that the transformed atlas roughly matches the position of the observed target data.

The string PIR means that, for the atlas image, as displayed in the notebook, the first row moves from anterior to posterior (P), the second row moves from superior to inferior (I), and the third row moves from left to right (R).  Often we cannot tell left from right, so I use a "right hand rule".

Similarly, the string LPS means that, for the target image, as displayed in the notebook, the first row moves from right to left (L, but see note above about "right hand rule"), the second row moves from anterior to posterior (P) and the third row moves from inferior to superior (S).

The notebook will generate several figures, and will save transformation results to the file specified as `outputname`.




= Step 3: transformation =

In this step we apply the saved transformations to new data

TODO