'''Unit testing for the emlddmm module.

This includes unit tests for methods in the emlddmm.py package.

Example
-------
Running the unit tests::

    python -m unittest test_emlddmm
    



'''

import unittest
import sys
sys.path.append('..')
import emlddmm

class TestIO(unittest.TestCase):
    '''
    '''
    def test_vtk_write_read(self):
        
        pass


if __name__ == '__main__':
    unittest.main()
