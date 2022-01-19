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

# # %%
# img_dir = '/home/brysongray/data/MD816/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
# xJ, J, _, _ = emlddmm.read_data(img_dir)
# J = J[0]
# print('max J: ', np.max(J), 'min J: ', np.min(J))
# dJ = [x[1]-x[0] for x in xJ]
# thresh = filters.threshold_otsu(J[int(J.shape[0]/2), :, :])
# J_verts, J_faces, J_normals, J_values = measure.marching_cubes(np.squeeze(np.array(J)), thresh, spacing=dJ)
# print('thresh: ', thresh)
# print('max values: ', np.max(J_values), 'min values: ', np.min(J_values))

# # %%
# # visualize surface
# surface_fig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(J_verts[:,0], J_verts[:,1], J_verts[:,2], J_faces, colormap='cool', opacity=0.5, figure=surface_fig)
# mlab.show()
# # %%
# import numpy as np


# fnames = ['a','b','c','d','e','f','g','h','i','j','k']
# idxs = np.arange(0,len(fnames),2)
# print('fnames: ', fnames)
# print('indices: ', idxs)
# print('new fnames: ', list(map(fnames.__getitem__, idxs)))
