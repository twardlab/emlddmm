import os
import re
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
import sys

subject_dir = 'C:/Users/BGAdmin/data/MD816/MD816_STIF'


def make_samples_tsv(subject_dir):
    # create list of files from source directory (subject_dir) in order of slice number
    fnames_ = os.listdir(subject_dir)
    fnames_ = [i for i in fnames_ if 'samples' not in i and 'fnames' not in i and '.json' not in i]
    fnames = sorted(fnames_, key=lambda x: x[-8:-4])

    # make samples.tsv file
    with open(os.path.join(subject_dir, 'samples.tsv'), 'w') as f:
        f.write('sample_id    participant_id    species    status\n')
        for i in range(len(fnames)):
            f.write('{0}    {1}    Mus Musculus    present\n'.format(fnames[i], fnames[i][:5]))


# Set up metadata JSON file
# Need Pixel Size, field of view, thickness, offset
def generate_sidecars(subject_dir, dtype='uint8', geometry_csv=None):
    # create list of image file names from samples.tsv
    try:
        samples = open(os.path.join(subject_dir,'samples.tsv')).read()
    except FileNotFoundError:
        print('No samples.tsv file was found in your subject directory.')
        sys.exit(1)
    fnames = [x.split('    ')[0] for x in samples.splitlines()][1:]
    sample_ids = [os.path.splitext(x)[0] for x in fnames]
    sizes = []
    space_directions = []
    space_origin = []
    dv = []
    # get geometry info
    if geometry_csv:
        geometry = pd.read_csv(geometry_csv, index_col=False)
        for i in range(len(geometry)):
            dv.append([geometry.iloc[i, 4], geometry.iloc[i, 5], geometry.iloc[i, 6]])
            sizes.append([4, geometry.iloc[i, 1], geometry.iloc[i, 2], geometry.iloc[i, 3]])
            space_directions.append([[dv[i][0], 0., 0.], [0., dv[i][1], 0.], [0., 0., 10.0]])
            space_origin.append([geometry.iloc[i, 7], geometry.iloc[i, 8], geometry.iloc[i, 9]])
    else:
        dx = 14.72
        dy = 14.72
        dz = 10
        z_range = np.arange(0, len(fnames), 1)
        z_range = dz * (z_range - z_range[-1] / 2)
        for i in range(len(fnames)):
            img_shape = plt.imread(os.path.join(subject_dir, fnames[i])).shape
            x_origin = -(img_shape[1] - 1) / 2 * dx
            y_origin = -(img_shape[0] - 1) / 2 * dy
            space_origin.append([x_origin, y_origin, z_range[i]])
            dv.append([dx, dy, dz])
            sizes.append([img_shape[2], img_shape[1], img_shape[0], 1])
            space_directions.append([[dx, 0., 0.], [0., dy, 0.], [0., 0., dz]])

    for i in range(len(sample_ids)):
        print(sample_ids[i] + '.json')
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
    make_samples_tsv(subject_dir)
    generate_sidecars(subject_dir)
