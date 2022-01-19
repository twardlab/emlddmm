import os
import re
import matplotlib
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage.transform import resize
from PIL import Image


def downsample_slices(subject_dir, output_dir, slice_downfactor=2, image_downfactor=2):
    try:
        fnames = os.listdir(subject_dir)
    except FileNotFoundError:
        print('subject directory does not exist')
        sys.exit(1)
    fnames = [i for i in fnames if 'samples' not in i and 'fnames' not in i and '.json' not in i]
    if len(fnames) == 0:
        print('subject directory is empty')
        sys.exit(1)
    fnames = sorted(fnames, key=lambda x: x.split('_')[-1][:4])
    # get the number of slices (missing slices included)
    max_slice = int(fnames[-1].split('_')[-1][:4])
    # get list of slices to be kept after downsampling (these are evenly spaced numbers that may include missing slices)
    slices_to_keep = np.arange(1, max_slice+1, slice_downfactor)
    # Find paths to existing slices in the downsampled list (intersection of existing slices and downsampled slices)
    slice_nums = [int(f.split('_')[-1][:4]) for f in fnames]
    slice_nums_down = [x for x in slices_to_keep if x in slice_nums]
    fnames_down = [f for f in fnames if int(f.split('_')[-1][:4]) in slice_nums_down]
    fpaths_down = [os.path.join(subject_dir,x) for x in fnames_down]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(fpaths_down)):
        if image_downfactor:
            img = plt.imread(fpaths_down[i])
            img_resized = resize(img, (img.shape[0] // image_downfactor, img.shape[1] // image_downfactor),\
                 anti_aliasing=True, preserve_range=False)
            plt.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(fpaths_down[i]))[0]+'.png'), img_resized)
        else:
            shutil.copy(fpaths_down[i], output_dir)


def make_samples_tsv(subject_dir, max_slice):
    # create list of files from source directory (subject_dir) in order of slice number
    try:
        fnames = os.listdir(subject_dir)
    except FileNotFoundError:
        print('subject directory does not exist')
        sys.exit(1)
    fnames = [i for i in fnames if 'samples' not in i and '.json' not in i]
    if len(fnames) == 0:
        print('subject directory is empty')
        sys.exit(1)
    fnames = sorted(fnames, key=lambda x: x.split('_')[-1][:4])

    # uncomment this block to include missing images in tsv file
    # # get missing image numbers
    # img_nums = np.array([int(x.split('_')[-1][:4]) for x in fnames])
    # slice_range = np.arange(1, max_slice+1)
    # missing_imgs = list(set(img_nums) - set(slice_range)) + list(set(slice_range) - set(img_nums))

    # print("missing images: \n", missing_imgs)

    # for i in range(len(missing_imgs)):
    #     prev_section_num = int(fnames[missing_imgs[i]-2].split('_')[-2])
    #     prev_slide_num = int(fnames[missing_imgs[i]-2].split('-')[1][1:])
    #     section_num = prev_section_num + 1 if prev_section_num != 3 else 1
    #     slide_num = prev_slide_num if prev_section_num != 3 else prev_slide_num + 1
    #     fname = fnames[0].split('-')[0] + f'-N{slide_num}-0000.00.00-00.00.00_' + fnames[0].split('_')[1]\
    #         + f'_{section_num}_' + str(missing_imgs[i]).zfill(4) + '.' + fnames[0].split('.')[-1]
    #     fnames.insert(missing_imgs[i]-1, fname)

    # make samples.tsv file
    with open(os.path.join(subject_dir, 'samples.tsv'), 'w') as f:
        f.write('sample_id    participant_id    species    status\n')
        status = 'present'
        # j = 0
        for i in range(len(fnames)):
        #     if j < len(missing_imgs) and missing_imgs[j] == i+1:
        #         status = 'missing'
        #         j += 1
        #     else:
        #         status = 'present' 
            f.write('{0}    {1}    Mus Musculus    {2}\n'.format(fnames[i], fnames[i][:5], status))


# Set up metadata JSON file
# Need Pixel Size, field of view, thickness, offset
def generate_sidecars(subject_dir, max_slice, dtype='uint8', dv=[14.72,14.72,10.], slice_downfactor=1, geometry_csv=None):
    try:
        fnames = os.listdir(subject_dir)
    except FileNotFoundError:
        print('subject directory does not exist')
        sys.exit(1)
    fnames = [i for i in fnames if 'samples' not in i and '.json' not in i]
    if len(fnames) == 0:
        print('subject directory is empty')
        sys.exit(1)
    fnames = sorted(fnames, key=lambda x: x.split('_')[-1][:4])
    sizes = []
    space_directions = []
    space_origin = []
    # get geometry info
    if geometry_csv:
        geometry = pd.read_csv(geometry_csv, index_col=False)
        dv = []
        for i in range(len(geometry)):
            dv.append([geometry.iloc[i, 4], geometry.iloc[i, 5], geometry.iloc[i, 6]])
            sizes.append([4, geometry.iloc[i, 1], geometry.iloc[i, 2], geometry.iloc[i, 3]])
            space_directions.append([[dv[i][0], 0., 0.], [0., dv[i][1], 0.], [0., 0., 10.0]])
            space_origin.append([geometry.iloc[i, 7], geometry.iloc[i, 8], geometry.iloc[i, 9]])
    else:
        dx, dy, dz = dv
        center = (max_slice - 1) / 2

        for i in range(len(fnames)):
            slice_num = int(fnames[i].split('_')[-1][:4])
            img_path = os.path.join(subject_dir, fnames[i])
            img_shape = plt.imread(img_path).shape
            x_origin = -(img_shape[1] - 1) / 2 * dx
            y_origin = -(img_shape[0] - 1) / 2 * dy
            space_origin.append([x_origin, y_origin, dz * (slice_num-1 - center)])
            sizes.append([img_shape[2], img_shape[1], img_shape[0], 1])
            space_directions.append([[dx, 0., 0.], [0., dy, 0.], [0., 0., dz*slice_downfactor]])

    sample_ids = [os.path.splitext(x)[0] for x in fnames]
    for i in range(len(fnames)):
        print(sample_ids[i] + '.json')
        img_path = os.path.join(subject_dir, fnames[i])
        with open(os.path.join(subject_dir, sample_ids[i] + '.json'), 'w') as f:
            f.write('{{\n'
                    '  \"DataFile\": \"{0}\",\n'
                    '  \"Type\": \"{1}\",\n'
                    '  \"Dimension\": 4,\n'
                    '  \"Sizes\": {2},\n'
                    '  \"Endian\": \"big\",\n'
                    '  \"Space\": \"inferior-right-posterior\",\n'
                    '  \"SpaceDimension\": 3,\n'
                    '  \"SpaceUnits\": [\"um\", \"um\", \"um\"],\n'
                    '  \"SpaceDirections\": ["none", {3}, {4}, {5}],\n'
                    '  \"SpaceOrigin\": {6}\n'
                    '}}'.format(fnames[i], dtype, sizes[i], space_directions[i][0], space_directions[i][1],
                                space_directions[i][2],
                                space_origin[i]))


if __name__ == '__main__':

    subject_dir = '/home/brysongray/data/MD816/MD816_STIF/'
    output_dir = '/home/brysongray/data/MD816_mini/MD816_STIF_mini_v2/'
    slice_down = 10
    res_down = 6
    max_slice = 1950
    dv = [14.72 * res_down, 14.72 * res_down, 10.]
    downsample_slices(subject_dir, output_dir, slice_downfactor=slice_down, image_downfactor=res_down)
    make_samples_tsv(output_dir, max_slice)
    generate_sidecars(output_dir, max_slice, dv=dv, slice_downfactor=slice_down)
