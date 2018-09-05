# dot_cython.pyx
import numpy as np
cimport numpy as np
cimport cython


cdef float MIN(cdef float x, cdef float y):
      return x if x <= y else y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int, ndim=1] farthest_point_sampling(np.ndarray[np.float32_t, ndim=2] points, K):
    cdef np.ndarray[np.float32_t, ndim=1] distance
    cdef int n, p, m
    cdef np.float32_t s
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    n, p, m = a.shape[0], a.shape[1], b.shape[1]
    c = np.zeros((n, m), dtype=np.float32)
    for i in xrange(n):
        for j in xrange(m):
            s = 0
            for k in xrange(p):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    return c

def _farthest_point_sampling(points, K):
    return farthest_point_sampling(points, K)