#pass in terminal: python setup.py build_ext --inplace
import numpy as np
cimport numpy as np
cimport cython # so we can use cython decorators


@cython.wraparound(False)
@cython.boundscheck(False)
def c_mult_cython_2d(np.ndarray[np.float64_t, ndim=2]  A, np.ndarray[np.float64_t, ndim=2]  B):
    cdef np.ndarray[np.float64_t, ndim=2] C
    cdef int a, b, c
    cdef int i, j, k
    cdef np.float64_t s
    a, b, c = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((a, c))
    for i in xrange(a):
        for j in xrange(c):
            s = 0
            for k in xrange(b):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

