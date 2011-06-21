import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
  double exp(double)
  double log(double)

ctypedef np.float64_t dtype_t

cdef double max2(double a, double b):
  if a > b :
    return a
  else :
    return b

cdef double min2(double a, double b):
  if a < b :
    return a
  else :
    return b

@cython.boundscheck(False)
def _logsum(int N, np.ndarray[dtype_t,ndim=1] X):
  cdef int i
  cdef double maxv,Xsum
  Xsum = 0.0
  #maxv = _maxv(N,X)
  maxv = X.max()
  for i in xrange(N):
    Xsum += exp(X[i] - maxv)
  return log(Xsum) + maxv

#@cython.boundscheck(False)
#cdef double _logsumexp(double x, double y):
#  cdef double min_v,max_v
#  min_v = min2(x,y)
#  max_v = max2(x,y)
#  if max_v - min_v > 50.0:
#    return max_v
#  else :
#    return max_v + log(exp(min_v-max_v) + 1.0)
#
#@cython.boundscheck(False)
#def _logsum(int N, np.ndarray[dtype_t,ndim=1] X):
#  cdef int i
#  cdef double Xsum
#  Xsum = -1.0e200
#  for i in xrange(N):
#    Xsum = _logsumexp(Xsum,X[i])
#  return Xsum

@cython.boundscheck(False)
def _logsum2d(int N,int M, np.ndarray[dtype_t,ndim=2] X):
  cdef int i
  cdef np.ndarray[dtype_t,ndim=1] Xsum = np.zeros(N)
  for i in xrange(N):
    Xsum[i] = _logsum(M,X[i])
  return Xsum
