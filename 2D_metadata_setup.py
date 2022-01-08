import os
import re
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
import sys


def downsample_slices(subject_dir, output_dir, downfactor=2):
    fnames = os.listdir(subject_dir)
    fnames = [i for i in fnames if 'samples' not in i and 'fnames' not in i and '.json' not in i]
    fnames = sorted(fnames, key=lambda x: x.split('_')[-1][:4])
    max_slice = int(fnames[-1].split('_')[-1][:4])
    idx_to_keep = np.arange(0, max_slice, downfactor)
    fnames_down = list(map(fnames.__getitem__, idx_to_keep))
    fpaths_down = [os.path.join(subject_dir,x) for x in fnames_down]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(fpaths_down[i], output_dir)

# def downsample_images(subject_dir, downfactor=2):


def make_samples_tsv(subject_dir):
    # create list of files from source directory (subject_dir) in order of slice number
    fnames = os.listdir(subject_dir)
    fnames = [i for i in fnames if 'samples' not in i and 'fnames' not in i and '.json' not in i]
    fnames = sorted(fnames, key=lambda x: x.split('_')[-1][:4])

    # get missing image numbers
    img_nums = np.array([int(x.split('_')[-1][:4]) for x in fnames])
    missing_imgs = []
    j = 0
    for i in range(np.size(img_nums)):
        if img_nums[j] == i+1:
            j += 1
        else:
            missing_imgs.append(i+1)
    print("missing images: \n", missing_imgs)

    for i in range(len(missing_imgs)):
        prev_section_num = int(fnames[missing_imgs[i]-2].split('_')[-2])
        prev_slide_num = int(fnames[missing_imgs[i]-2].split('-')[1][1:])
        section_num = prev_section_num + 1 if prev_section_num != 3 else 1
        slide_num = prev_slide_num if prev_section_num != 3 else prev_slide_num + 1
        fname = fnames[0].split('-')[0] + f'-N{slide_num}-0000.00.00-00.00.00_' + fnames[0].split('_')[1]\
            + f'_{section_num}_' + str(missing_imgs[i]).zfill(4) + '.' + fnames[0].split('.')[-1]
        fnames.insert(missing_imgs[i]-1, fname)

    # make samples.tsv file
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    with open(os.path.join(subject_dir, 'samples.tsv'), 'w') as f:
        f.write('sample_id    participant_id    species    status\n')
        j = 0
        for i in range(len(fnames)):
            if j < len(missing_imgs) and missing_imgs[j] == i+1:
                status = 'missing'
                j += 1
            else:
                status = 'present' 
            f.write('{0}    {1}    Mus Musculus    {2}\n'.format(fnames[i], fnames[i][:5], status))


# Set up metadata JSON file
# Need Pixel Size, field of view, thickness, offset
def generate_sidecars(subject_dir, dtype='uint8', dv=[14.72,14.72,10.], geometry_csv=None):
    # create list of image file names from samples.tsv
    try:
        samples = open(os.path.join(subject_dir,'samples.tsv')).read()
    except FileNotFoundError:
        print('No samples.tsv file was found in your subject directory.')
        sys.exit(1)
    samples = samples.splitlines()[1:]
    if '\t' in samples[0]:
        fnames = [x.split('\t')[0] for x in samples]
    else:
        fnames = [x.split('    ')[0] for x in samples]
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
        max_slice = fnames[-1].split('_')[-1][:4]
        z_range = np.arange(0, max_slice, 1)
        z_range = dz * (z_range - z_range[-1] / 2)
        for i in range(len(fnames)):
            img_path = os.path.join(subject_dir, fnames[i])
            if os.path.exists(img_path): # not all samples are present
                img_shape = plt.imread(img_path).shape
                x_origin = -(img_shape[1] - 1) / 2 * dx
                y_origin = -(img_shape[0] - 1) / 2 * dy
                space_origin.append([x_origin, y_origin, z_range[i]])
                sizes.append([img_shape[2], img_shape[1], img_shape[0], 1])
                space_directions.append([[dx, 0., 0.], [0., dy, 0.], [0., 0., dz]])

    sample_ids = [os.path.splitext(x)[0] for x in fnames]
    for i in range(len(fnames)):
        print(sample_ids[i] + '.json')
        img_path = os.path.join(subject_dir, fnames[i])
        if os.path.exists(img_path):
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

    subject_dir = 'C:/Users/BGAdmin/data/MD816/MD816_STIF/'
    output_dir = 'C:/Users/BGAdmin/data/MD816_mini/MD816_STIF_mini/'

    downsample_slices(subject_dir, output_dir, downfactor=4)
    make_samples_tsv(output_dir)
    generate_sidecars(output_dir)
