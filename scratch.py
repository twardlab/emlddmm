# %%
# FINDING OUTLIER IMAGE SLICES
#####################################################################################################################
# destination = 'C:\\Users\\BGAdmin\\data\\MD816\\outliers'
# file_list = os.listdir(target_name)
# height = []
# width = []
# for j in file_list:
#     if '.tif' in j:
#         J = plt.imread(os.path.join(target_name,j))
#         height.append(J.shape[0])
#         width.append(J.shape[1])
#         ratio = J.shape[1]/J.shape[0]
#         shape_sum = J.shape[1]+J.shape[0]
#         if shape_sum >= 2308:
#             print('size > 99th:\n',j)
#             fname, ext = os.path.splitext(j)
#             plt.imsave(fname=os.path.join(destination, 'large', fname+'.png'), arr=J, format='png')
#         if ratio >= 1.66750:
#             print('ratio > 99th:\n', j)
#             plt.imsave(fname=os.path.join(destination, 'wide', fname+'.png'), arr=J, format='png')
#         if ratio <= 0.75074:
#             print('ratio < 1st:\n', j)
#             plt.imsave(fname=os.path.join(destination, 'tall', fname+'.png'), arr=J, format='png')


# %%
# size_ratio = np.array(width)/np.array(height)
# size_sum = np.array(width) + np.array(height)
# ratio_99 = np.percentile(size_ratio, 99)
# ratio_1 = np.percentile(size_ratio, 1)
# sum_99 = np.percentile(size_sum, 99)
# width_99 = np.percentile(width,99)
# height_99 = np.percentile(height,99)
# print('width 99th percentile: ', width_99)
# print('height 99th percentile: ', height_99)
# print('sum 99th percentile: ', sum_99)
# print('ratio 98% interval: ', ratio_1, ratio_99)

# %%
# fig, axs = plt.subplots(3)
# fig.set_size_inches(10,10)
# size_ratio = np.array(width)/np.array(height)
# axs[0].hist(size_ratio, bins=50)
# axs[0].set_title('Size Ratio (width/height)')
# axs[1].hist(height, bins=50)
# axs[1].set_title('Height')
# axs[2].hist(width, bins=50)
# axs[2].set_title('Width')
# plt.savefig(os.path.join(destination, 'histograms.png'))
# plt.show()

# %%
# import emlddmm
# from skimage import filters, measure
# import numpy as np
# from mayavi import mlab
# # from medpy.metric import binary
# import matplotlib
# matplotlib.use('qtagg')


# img = 'C:\\Users\\BGAdmin\\data\\MD816/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
# out = 'C:\\Users\\BGAdmin\\emlddmm\\test_outputs'

# xI,I,_,_ = emlddmm.read_data(img)

# fig = emlddmm.draw(I,xI)
# fig[0].suptitle('image')
# fig[0].canvas.draw()

# dI = [x[1]-x[0] for x in xI]
# thresh = filters.threshold_otsu(np.array(I)[0,int(I.shape[1]/2),:,:])
# # AphiI_verts, AphiI_faces, AphiI_normals, AphiI_values  = measure.marching_cubes(np.squeeze(np.array(AphiI)), thresh, spacing=dI)
# I_verts, I_faces, I_normals, I_values = measure.marching_cubes(np.squeeze(np.array(I)), thresh, spacing=dI)

# surface_fig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# # mlab.triangular_mesh(AphiI_verts[:,0], AphiI_verts[:,1], AphiI_verts[:,2], AphiI_faces, colormap='hot', opacity=0.5, figure=surface_fig)
# mlab.triangular_mesh(I_verts[:,0], I_verts[:,1], I_verts[:,2], I_faces, colormap='cool', opacity=0.5, figure=surface_fig)
# mlab.show()
# mlab.savefig(out+'surfaces.obj')
# mlab.close()

# # %%
# import numpy as np
# from skimage import measure, color, filters
# import emlddmm
# import matplotlib.pyplot as plt
# from mayavi import mlab


# img_dir = 'MD787_small_nissl'

# # visualize 3d mesh of 2d reconstructed volume
# xJ, J, _, _ = emlddmm.read_data(img_dir)
# mask = J[3,...]
# J_gray = color.rgb2gray(np.transpose(J[:3, ...]*mask, (1,2,3,0)))

# dJ = [x[1]-x[0] for x in xJ]
# thresh = filters.threshold_otsu(J_gray[int(J_gray.shape[0]/2), 50:250, 10:300])
# J_verts, J_faces, J_normals, J_values = measure.marching_cubes(np.squeeze(np.array(J_gray)), thresh, spacing=dJ)

# surface_fig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(J_verts[:,0], J_verts[:,1], J_verts[:,2], J_faces, colormap='cool', opacity=0.5, figure=surface_fig)
# mlab.show()


# # %%

# idx = int(J_gray.shape[0]/2)
# J_slice = J_gray[idx, 50:250, 10:300]  # manually cropping image to exclude padding
# plt.figure()
# plt.imshow(J_slice, cmap = plt.cm.gray)

# thresh = filters.threshold_otsu(J_slice)

# contours = measure.find_contours(J_slice, thresh)

# # manually get the right contour
# contours_len = [len(c) for c in contours]
# contours_len[40] = 0
# contours_len_max_idx = np.argmax(contours_len)
# print(contours_len_max_idx)
# print(contours_len[contours_len_max_idx])
# # Display the image and plot all contours found
# fig, ax = plt.subplots()
# ax.imshow(J_slice, cmap=plt.cm.gray)

# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

# fig, ax = plt.subplots()
# ax.imshow(J_slice, cmap=plt.cm.gray)   
# ax.plot(contours[contours_len_max_idx][:, 1], contours[contours_len_max_idx][:, 0], linewidth=2)

# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()

# # %%
# # display binarized image using otsu's method

# idx = int(J_gray.shape[0]/2)
# J_slice = J_gray[idx, 50:250, 10:300]  # manually cropping image to exclude padding
# plt.figure()
# plt.imshow(J_slice, cmap = plt.cm.gray)

# thresh = filters.threshold_otsu(J_slice)
# # binarize each image
# J_slice_bool = (J_slice > thresh) * 1
# plt.figure()
# plt.imshow(J_slice_bool)

# # %%
# from skimage.filters import gaussian
# from skimage.segmentation import active_contour

# def get_border(img):
# # input: 2d image array
#     h = img.shape[0]-11
#     w = img.shape[1]-11
#     s1 = np.array([np.linspace(10,h,20), np.ones(20)*10]).T
#     s2 = np.array([(h)*np.ones(20), np.linspace(10,w,20)]).T[1:]
#     s3 = np.array([np.linspace(10,h,20)[::-1], w*np.ones(20)]).T[1:]
#     s4 = np.array([np.ones(20)*10, np.linspace(10,w,20)[::-1]]).T[1:]

#     return np.concatenate((s1,s2,s3,s4))

# img_dir = 'MD787_small_nissl'
# xJ, J, _, _ = emlddmm.read_data(img_dir)
# J_gray = color.rgb2gray(np.transpose(J[:3, ...], (1,2,3,0)))
# center = [int(J.shape[1]/2), int(J.shape[2]/2), int(J.shape[3]/2)]

# # plt.figure()
# # plt.imshow(J_gray[center[0]], cmap='gray')
# # plt.show()
# # b = get_border(J_gray[center[0]])

# w = []
# h = []
# W = []
# H = []
# J_cropped = []
# init = []
# snake = []
# for i in range(J.shape[1]):  # for each slice
#     row = np.where(mask[i,center[1],:]==True)
#     w.append(np.min(row))
#     W.append(np.max(row))
#     col = np.where(mask[i,:,center[2]]==True)
#     h.append(np.min(col))
#     H.append(np.max(col))
#     J_cropped.append(J_gray[i, h[i]:H[i], w[i]:W[i]])
#     init.append(get_border(J_cropped[i]))
#     snake.append( active_contour( J_cropped[i], #gaussian(J_cropped[i], 1, preserve_range=False),
#                                   init[i], boundary_condition='periodic',
#                                   alpha=0.0015, beta=10.0, w_line=-0.1, w_edge=2, gamma=0.001))
#     print(f'finished slice {i}')

# img = J_cropped[center[0]]
# fig, ax = plt.subplots(figsize=(9, 5))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[center[0]][:, 1], init[center[0]][:, 0], '--r', lw=3)
# ax.plot(snake[center[0]][:, 1], snake[center[0]][:, 0], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])

# plt.show()



# # %%
# plt.figure(figsize=(9, 5))
# plt.imshow(J_cropped[center[0]],cmap='gray')
# plt.show()


# %%
import numpy as np
import emlddmm
import torch
import matplotlib.pyplot as plt

MR_img = '/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
CCF_img = '/home/brysongray/data/MD816_mini/average_template_50.vtk'
CCFtoMRI_disp = '/home/brysongray/emlddmm/transformation_graph_outputs/CCF/MRI_to_CCF/transforms/CCF_to_MRI_displacement.vtk'
# velocity = '/home/brysongray/emlddmm/transformation_graph_outputs/CCF/MRItoCCF/transforms/velocity.vtk'

xJ,J,title,names = emlddmm.read_data(MR_img)
xI, I, title, names = emlddmm.read_data(CCF_img)

x, disp, title, names = emlddmm.read_vtk_data(CCFtoMRI_disp)
disp = torch.as_tensor(disp)

down = [a//b for a, b in zip(I.shape[1:], disp.shape[2:])]

xI,I = emlddmm.downsample_image_domain(xI,I,down)
XI = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))

X = disp[0] + XI

AphiI = emlddmm.apply_transform_float(xJ, J, X)
print(AphiI.shape)
src='MRI'
dest='CCF'
fig = emlddmm.draw(AphiI, xI)
fig[0].suptitle('transformed {src} to {dest}'.format(src=src, dest=dest))
fig[0].canvas.draw()
plt.show()

# %%
import emlddmm

MRItoCCF = '/home/brysongray/emlddmm/transformation_graph_outputs/CCF/MRI_to_CCF/images/MRI_masked_to_CCF.vtk'
CCFtoMRI = '/home/brysongray/emlddmm/transformation_graph_outputs/MRI/CCF_to_MRI/images/CCF_average_template_50_to_MRI.vtk'
HISTtoMRI = '/home/brysongray/emlddmm/transformation_graph_outputs/MRI/HIST_to_MRI/images/HIST_nissl_to_MRI.vtk'
xI, I, _,_ = emlddmm.read_data(MRItoCCF)
xJ, J, _,_ = emlddmm.read_data(HISTtoMRI)
print(J.shape)
print([len(x) for x in xJ])
#%%
# emlddmm.draw(I,xI)
print(J.shape)
print(J[:,256,None,...].shape)
xJ_ = [np.array([xJ[0][256]]), xJ[1], xJ[2]]

emlddmm.draw(J[:,256,None, ...], xJ_)
plt.show()

# %%
# in the special case of transforming an image series to registered or input space, the dest will be a directory containing 2d Affines
if dest_path == f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED':
    xJ, J, J_title, _ = emlddmm.read_data(src_path) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    x_series = [torch.as_tensor(x,dtype=dtype,device=device) for x in xJ]
    X_series = torch.stack(torch.meshgrid(x_series), -1)
    transforms_ls = os.listdir(os.path.join(out, dest_path))
    transforms_ls = sorted(transforms_ls, key=lambda x: x.split('_matrix.txt')[0][-4:])

    A2d = []
    for t in transforms_ls:
        A2d_ = np.genfromtxt(os.path.join(out, dest_path, t), delimiter=',')
        # note that there are nans at the end if I have commas at the end
        if np.isnan(A2d_[0, -1]):
            A2d_ = A2d_[:, :A2d_.shape[1] - 1]
        A2d.append(A2d_)

    A2d = torch.as_tensor(np.stack(A2d),dtype=dtype,device=device)
    A2di = torch.inverse(A2d)
    points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] # reconstructed space needs to be created from the 2d series coordinates
    m0 = torch.min(points[..., 0])
    M0 = torch.max(points[..., 0])
    m1 = torch.min(points[..., 1])
    M1 = torch.max(points[..., 1])
    # construct a recon domain
    dJ = [x[1] - x[0] for x in x_series]
    # print('dJ shape: ', [x.shape for x in dJ])
    xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
    xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
    xr = x_series[0], xr0, xr1
    XR = torch.stack(torch.meshgrid(xr), -1)
    # reconstruct 2d series
    Xs = torch.clone(XR)
    Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
    Xs = Xs.permute(3, 0, 1, 2)
    Jr = emlddmm.interp(xJ, J, Xs)

    # write out displacement
    input_disp = (Xs - X_series.permute(3,0,1,2)).cpu()[None]
    for i in range(input_disp.shape[2]):
        x_series_ = [x_series[0][i], x_series[1], x_series[2]]
        x_series_[0] = torch.tensor([x_series_[0], x_series_[0] + 10])
        xr_ = [xr[0][i], xr[1], xr[2]]
        xr_[0] = torch.tensor([xr_[0], xr_[0] + 10])
        # write out input to dest displacement
        input_dir = os.path.join(out, f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED')
        output_name = os.path.join(input_dir, transforms_ls[i].split('_matrix.txt')[0]+'_displacement.vtk')
        title = transforms_ls[i].split('_matrix.txt')[0]+'_displacement'
        emlddmm.write_vtk_data(output_name, x_series_, input_disp[:,:, i, None, ...], title)

    # write out image
    fig = emlddmm.draw(Jr, xr)
    fig[0].suptitle(f'transformed {src_space} {src_img} to {dest_space} REGISTERED')
    fig[0].canvas.draw()
    # save transformed 3d image   
    img_out = os.path.join(out, f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED')
    if not os.path.exists(img_out):
        os.makedirs(img_out)
    emlddmm.write_vtk_data(os.path.join(img_out, f'{src_space}_INPUT_to_{dest_space}_REGISTERED.vtk'), xr, Jr, f'{src_space}_INPUT_to_{dest_space}_REGISTERED')

#%%
# transform images using disp fields
def transform_img(adj, spaces, src, dest, out, src_img='', dest_img=''):
    '''
    Parameters
    ----------
    adj: adjacency list
        of the form: [{1: ('path to transformation from space 0 to space 1', 'b'}, {0: ('path to transform from space 0 to space 1', 'f') }]
    spaces: spaces dict
        example: {'MRI':0, 'CT':1, 'ATLAS':2}
    src: source space (str)
    dest: destination space (str)
    out: output directory
    src_img: path to source image (image to be transformed)
    dest_img: path to destination image (image in space to which source image will be matched)
    
    Returns
    ----------
    x: List of arrays storing voxel locations
    AphiI: Transformed image as tensor

    input: path to image to be transformed (src_img or I), img space to to which the source image will be matched (dest_img, J),
     adjacency list and spaces dict from run_registration, source and destination space names
    return: x, transfromed image
    '''
    # get transformation sequence
    path = findShortestPath(adj, spaces[src], spaces[dest], len(spaces))
    if len(path) < 2:
        return
    print("\nPath is:")

    for i in path:
        for key, value in spaces.items():
            if i == value:
                print(key, end=' ')

    transformation_seq = getTransformation(adj, path)
    print('\nTransformation sequence: ', transformation_seq)

    # load source and destination images
    xI, I, I_title, _ = emlddmm.read_data(dest_img) # the space to transform into
    I = I.astype(float)
    I = torch.as_tensor(I, dtype=dtype, device=device)
    xI = [torch.as_tensor(x,dtype=dtype,device=device) for x in xI]
    xJ, J, J_title, _ = emlddmm.read_data(src_img) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    xJ = [torch.as_tensor(x,dtype=dtype,device=device) for x in xJ]

    # compose displacements
    # TODO
    # first interpolate all displacements in sequence into destination space,
    # then add them together
    disp_list = []
    for i in reversed(range(len(transformation_seq))):
        # path = glob.glob(transformation_seq[i] + '/transforms/*displacement.vtk')
        x, disp, title, names = emlddmm.read_vtk_data(transformation_seq[i])
        if i == len(transformation_seq)-1: # if the first displacement in the sequence
            down = [a//b for a, b in zip(I.shape[1:], disp.shape[2:])] # check if displacement image was downsampled from original
            xI,I = emlddmm.downsample_image_domain(xI,I,down) # downsample the domain
            XI = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))
            disp_list.append(torch.as_tensor(disp)) # add the displacement
        else: # otherwise we need to interpolate to the destination space
            ID = torch.stack(torch.meshgrid(x))[None]
            disp = emlddmm.interp(x,(disp - ID), XI) + XI
            disp_list.append(torch.as_tensor(disp))
    # sum the displacements
    # v = torch.cat(disp_list)
    disp = torch.sum(torch.stack(disp_list),0).to(device=device)
    print('disp shape: ', disp.shape)
    print('XI shape: ', XI.shape)
    X = disp[0] + XI

    AphiI = emlddmm.apply_transform_float(xJ, J, X)

    # save figure and text file of transformation order
    if not os.path.exists(out):
        os.makedirs(out)
    # plt.savefig(os.path.join(out, '{src}_{dest}'.format(src=src, dest=dest))) # TODO: this image will be saved in qc
    with open(os.path.join(out, '{src}_{dest}.txt'.format(src=src, dest=dest)), 'w') as f:
        for transform in reversed(transformation_seq[1:]):
            f.write(str(transform) + ', ')
        f.write(str(transformation_seq[0]))

    return xI, AphiI

