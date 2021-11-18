import emlddmm
from mayavi import mlab
from skimage import measure, filters
import numpy as np


img = 'C:\\Users\\BGAdmin\\data\\MD816/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
output_dir = 'C:\\Users\\BGAdmin\\emlddmm\\test_outputs'

xI, I, _, _ = emlddmm.read_data(img)
# fig = emlddmm.draw(I,xI)
# fig[0].suptitle('test')
# fig[0].canvas.draw()

dI = [x[1]-x[0] for x in xI]
thresh = filters.threshold_otsu(np.array(I)[0,int(I.shape[1]/2),:,:])
# AphiI_verts, AphiI_faces, AphiI_normals, AphiI_values  = measure.marching_cubes(np.squeeze(np.array(AphiI)), thresh, spacing=dI)
I_verts, I_faces, I_normals, I_values = measure.marching_cubes(np.squeeze(np.array(I)), thresh, spacing=dI)

surface_fig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(AphiI_verts[:,0], AphiI_verts[:,1], AphiI_verts[:,2], AphiI_faces, colormap='hot', opacity=0.5, figure=surface_fig)
mlab.triangular_mesh(I_verts[:,0], I_verts[:,1], I_verts[:,2], I_faces, colormap='cool', opacity=0.5, figure=surface_fig)
mlab.show()
mlab.savefig(output_dir+'/surfaces.obj')
mlab.close()


