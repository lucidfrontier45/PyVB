import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t dtype_t

@cython.boundscheck(False)
def _evaluateHiddenState_C( np.ndarray[dtype_t, ndim = 2] z,\
    np.ndarray[dtype_t, ndim = 2] obs,\
    np.ndarray[dtype_t, ndim = 1] nu, \
    int nobs,int nmix, np.ndarray[dtype_t, ndim = 2] m,\
    np.ndarray[dtype_t, ndim = 3] S):
  cdef int i,j,n,k,dim
  cdef double temp
  cdef np.ndarray[dtype_t, ndim = 2] dobs
  dim = len(obs[0])
  dobs = np.zeros(nobs*dim).reshape(nobs,dim)
  for k in xrange(nmix):
    #dobs = obs - m[k]
    for n in xrange(nobs):
      for i in xrange(dim):
        dobs[n,i] = obs[n,i] - m[k,i]
      temp = 0.0
      for i in xrange(dim):
        for j in xrange(dim):
          temp -=  dobs[n,i] * S[k,i,j] * dobs[n,j]
      temp = temp * 0.5 * nu[k]
      z[n,k] += temp

@cython.boundscheck(False)
def _lnPD_C( np.ndarray[dtype_t, ndim = 2] z,\
    np.ndarray[dtype_t, ndim = 2] obs,\
    int nobs,int nmix, np.ndarray[dtype_t, ndim = 2] m,\
    np.ndarray[dtype_t, ndim = 3] eS):
  """ 
  acceralate computaing lnPD in _VBLowerBound
  equivalent to but faster than
  for k in xxrange(nmix):
    dobs = obs - self.m[k]
    lnPD += np.dot(self.z[:,k],np.diag(np.dot(dobs,np.dot(self._eS[k],dobs.T))))
  """
  cdef int i,j,n,k,dim
  cdef double lnPD = 0.0, temp_ln
  cdef np.ndarray[dtype_t, ndim = 2] dobs
  dim = len(obs[0])
  dobs = np.zeros(nobs*dim).reshape(nobs,dim)
  for k in xrange(nmix):
    #dobs = obs - m[k]
    for n in xrange(nobs):
      temp_ln = 0.0
      for i in xrange(dim):
        dobs[n,i] = obs[n,i] - m[k,i]
      for i in xrange(dim):
        for j in xrange(dim):
          temp_ln += dobs[n,i] * eS[k,i,j] * dobs[n,j]
      lnPD += z[n,k] * temp_ln
  return lnPD
