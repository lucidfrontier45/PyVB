import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
  double exp(double)
  double log(double)

ctypedef np.float64_t dtype_t

@cython.boundscheck(False)
def _logsum(int N, np.ndarray[dtype_t,ndim=1] X):
  cdef int i
  cdef double maxv,Xsum
  Xsum = 0.0
  maxv = X.max()
  for i in xrange(N):
    Xsum += exp(X[i] - maxv)
  return log(Xsum) + maxv

@cython.boundscheck(False)
def _forward_C(int T, int N,\
    np.ndarray[dtype_t,ndim=1] lnpi,\
    np.ndarray[dtype_t,ndim=2] lnA,\
    np.ndarray[dtype_t,ndim=2] lnf, \
    np.ndarray[dtype_t,ndim=2] lnalpha):

  cdef int t,i,j
  cdef double lnP
  cdef np.ndarray[dtype_t,ndim=1] temp
  temp = np.zeros(N)

  for i in xrange(N):
    lnalpha[0,i] = lnpi[i] + lnf[0,i]

  for t in xrange(1,T):
    for j in xrange(N):
      for i in xrange(N):
        temp[i] = lnalpha[t-1,i] + lnA[i,j]
      lnalpha[t,j] = _logsum(N,temp) + lnf[t,j]

@cython.boundscheck(False)
def _backward_C(int T, int N,\
    np.ndarray[dtype_t,ndim=1] lnpi,\
    np.ndarray[dtype_t,ndim=2] lnA,\
    np.ndarray[dtype_t,ndim=2] lnf, \
    np.ndarray[dtype_t,ndim=2] lnbeta):

  cdef int t,i,j
  cdef double lnP
  cdef np.ndarray[dtype_t,ndim=1] temp
  temp = np.zeros(N)

  for i in xrange(N):
    lnbeta[T-1,i] = 0.0

  for t in xrange(T-2,-1,-1):
    for i in xrange(N):
      for j in xrange(N):
        temp[j] = lnA[i,j] + lnf[t+1,j] + lnbeta[t+1,j]
      lnbeta[t,i] = _logsum(N,temp)

@cython.boundscheck(False)
def _compute_lnEta_C(int T, int N,\
    np.ndarray[dtype_t,ndim=2] lnalpha,\
    np.ndarray[dtype_t,ndim=2] lnA,\
    np.ndarray[dtype_t,ndim=2] lnbeta,\
    np.ndarray[dtype_t,ndim=2] lnf, \
    double lnP_f, \
    np.ndarray[dtype_t,ndim=3] lneta):

  cdef int i,j,t
  for t in xrange(T-1):
    for i in xrange(N):
      for j in xrange(N):
        lneta[t,i,j] = lnalpha[t,i]+lnA[i,j]+lnf[t+1,j]+lnbeta[t+1,j]-lnP_f

