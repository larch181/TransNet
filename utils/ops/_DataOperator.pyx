""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "DataOperator.h":
   void farthest_point_sampling(double * points, int * out_index, int size, int K)

# create the wrapper code, with numpy type annotations
def farthest_point_samplin_func(np.ndarray[double, ndim=2, mode="c"] points not None,
                     np.ndarray[int, ndim=1, mode="c"] out_index not None, K):
    # farthest_point_sampling(<double*> np.PyArray_DATA(points),
    #             <int*> np.PyArray_DATA(out_index),
    #             points.shape[0],K)
    farthest_point_sampling(<double*> &points[0,0], <int*> &out_index[0],
                points.shape[0],K)
