'''Unit testing for the emlddmm module.

This includes unit tests for methods in the emlddmm.py package.

Example
-------
Running the unit tests::

    pytest test_emlddmm.py
    



'''

import pytest
import sys
sys.path.insert(0, '/home/brysongray/emlddmm')
import emlddmm
# from ... import emlddmm
import numpy as np
import os
import time


def ellipsoid():
    ni = 100
    nj = 120
    nk = 110
    xI = [np.arange(ni)-(ni-1)/2,np.arange(nj)-(nj-1)/2,np.arange(nk)-(nk-1)/2]
    XI = np.stack(np.meshgrid(xI[0],xI[1],xI[2], indexing='ij'))
    condition = lambda x,a,b,c : x[0]**2 / a**2 + x[1]**2 / b**2 + x[2]**2 / c**2
    a = 15
    b = 30
    c = 20
    I = np.where(condition(XI,a,b,c) < 1.0, 1.0, 0.0)[None]

    return xI, I


class TestIO:
    ''' Read and write data test

    '''
    # construct binary 3D ellipsoid
    fname = '/home/brysongray/emlddmm/tests/ellipsoid1'
    title = 'ellipsoid1'
    xI, I = ellipsoid()

    def test_write_vtk_data(self):
        # write out image in vtk format
        emlddmm.write_vtk_data(self.fname+'.vtk', self.xI, self.I, self.title)
        # assert that the file is modified
        writetime = os.path.getmtime(self.fname+'.vtk')
        assert round(writetime, 0)==round(time.time(),0)


    def test_read_vtk_data(self):
        _,J,_,_ = emlddmm.read_vtk_data(self.fname+'.vtk')
        assert np.allclose(J,self.I)


    def test_write_data(self):
        # write out image in nifti format
        emlddmm.write_data(self.fname+'.nii', self.xI, self.I, self.title)
        # assert that the file is modified
        writetime = os.path.getmtime(self.fname+'.nii')
        assert round(writetime, 0)==round(time.time(),0)


    def test_read_data(self):
        _,J,_,_ = emlddmm.read_data(self.fname+'.nii')
        assert np.allclose(J,self.I)


class TestEmlddmm:

    def test_emlddmm(self):

        pass

    def test_emlddmm_multiscale(self):

        pass