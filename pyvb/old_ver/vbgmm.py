#!/usr/bin/python

import numpy as np
from numpy.random import randn,dirichlet
from scipy.linalg import det, inv
from scipy.cluster import vq
from scipy.special import psi,gammaln
from core import normalize

try:
  from _vbgmm import _evaluateHiddenState_C, _lnPD_C
  ext_imported = True
except:
  ext_imported = False
  print "warning, Cython extension module was not found"
  print "computation can be slower"

def testData1(n=100):
  C = np.array([[0., -0.7], [3.5, .7]])
  X = randn(n, 2) + np.array([2, 2])
  return X

def testData2(n=100):
  C = np.array([[0., -0.7], [3.5, .7]])
  X = np.r_[np.dot(randn(n, 2), C) + np.array([1,2]), randn(n*2, 2) + np.array([3, 3])]
  #X = np.r_[randn(n, 2), randn(n*2, 2) + np.array([5, 5])]
  return X

def readObs(obs):
  data = np.array(obs)
  shape = data.shape
  if len(shape) == 1:
    shape = (1,shape[0])
    data.shape = shape

  return data

class VBGMM:
  def __init__(self,nmix=10,alpha=0.0,beta=0.1,gamma=1,delta=10.0):
    self._nstates = nmix
    # hyper-parameters for mean vectors
    self._alpha = alpha 
    self._beta = beta 
    # hyper-parameters for precision matrix 
    self._gamma = gamma
    self._delta = delta

  def _init_params(self,obs,use_emgmm=False):
    nobs,ndim = obs.shape
    nmix = self._nstates
    self._set_prior(obs)
    self._set_posterior(obs,use_emgmm)
    #self._emu = self.m
    #self._emu2 = np.array(self.C)
    self._eS = np.tile(self._beta*np.identity(ndim),(nmix,1,1))
    self._elnS = np.zeros(nmix)

  def _set_prior(self,obs):
    nobs,ndim = obs.shape
    nmix = self._nstates
    # prior mean vector
    if self._alpha > 0.0:
      self._alpha = randn(ndim) * self._alpha
    else:
      self._alpha = None
    # prior precision matrix
    self._B = self._beta * np.identity(ndim)
    # prior degree of freedom
    self._gamma += ndim
    # prior scale matrix
    self._D = self._delta * np.identity(ndim)
    
  def _set_posterior(self,obs,use_emgmm=False):
    nobs,ndim = obs.shape
    nmix = self._nstates
    # hidden states
    self.z = dirichlet(np.tile(1.0/nmix,nmix),nobs)
    # mixing coefficients
    #self.pi = dirichlet(np.tile(1.0/nmix,nmix))
    self.pi = self.z.sum(0)
    # posterior mean vector
    self.m, temp = vq.kmeans2(obs,nmix)
    # posterior covarience matrix
    self.C = np.tile(np.identity(ndim) * self._delta,(nmix,1,1))
    # posterior degree of freedom
    self.nu = np.tile(float(nobs)/nmix,nmix)
    # posterior scale matrix
    self.V = np.tile(np.array(self._D),(nmix,1,1))
    if use_emgmm:
      try:
        from scikits.learn.mixture import GMM
        print "initialize posterior parameter by EMGMM"
        emgmm = GMM(nmix,"full")
        emgmm.fit(obs,50)
        self.z = emgmm.predict_proba(obs)
        self.pi = emgmm._get_weights()
        self.m = np.array(emgmm._get_means())
        self.nu = self.z.sum(0)
      except ImportError:
        print "warning couldn't import scikits.learn.mixture"
        pass

  def _hyper_parameters(self):
    print "alpha",self._alpha
    print "beta",self._B
    print "gamma",self._gamma
    print "delta",self._D

  def _posterior_parameters(self):
    print "m",self.m
    print "C",self.C
    print "nu",self.nu
    print "V",self.V
    print "pi",self.pi

  def _debug(self):
    self._hyper_parameters()
    self._posterior_parameters()
    print "z",self.z


  def _calcSufficientStatistic(self,obs):
    nobs,ndim = obs.shape
    nmix = self._nstates
    #self._emu = self.m
    for k in xrange(nmix):
      #self._emu2[k] = self.C[k] + np.outer(self.m[k],self.m[k])
      self._eS[k]= self.nu[k] * inv(self.V[k])
      self._elnS[k] = psi(np.arange(self.nu[k]+1-ndim,self.nu[k]+1)/2.0).sum() \
          - np.log(det(self.V[k]))
    self._elnS += ndim * np.log(2)

  def _evaluateHiddenState(self,obs,use_ext=True):
    nobs,ndim = obs.shape
    nmix = self._nstates
    z = np.tile(0.5 * self._elnS + np.log(self.pi),(nobs,1))
    if use_ext and ext_imported :
      for k in xrange(nmix):
        z[:,k] -= 0.5 * (np.trace(np.dot(self._eS[k],self.C[k])))
      _evaluateHiddenState_C(z,obs,nobs,nmix,self.m,self._eS)
    else :
      for k in xrange(nmix):
        # very slow! need Fortran or C codes
        dobs = obs - self.m[k]
        z[:,k] -= 0.5 * (np.trace(np.dot(self._eS[k],self.C[k])) + \
            np.diag(np.dot(np.dot(dobs,self._eS[k]),dobs.T)))
        #for n in xrange(nobs):
        #  dobs = obs[n] - self.m[k]
        #  cv = self.C[k] + np.outer(dobs,dobs)
        #  z[n,k] -= 0.5 * np.trace(np.dot(self._eS[k],cv))

    z = z - z.max(1)[np.newaxis].T
    z = np.exp(z)
    z = normalize(z,1)
    #z = z.round(5)
    #z = normalize(z,1)
    return z

  def _updatePosteriorParameters(self,obs):
    nobs,ndim = obs.shape
    nmix = self._nstates
    zsum = self.z.sum(0)
    mm = np.dot(self.z.T,obs)
    self.pi = zsum / float(nobs)
    self.nu = self._gamma + zsum
    for k in xrange(nmix):
      dobs = obs - self.m[k]
      self.C[k] = inv(self._B + self._eS[k] * zsum[k])
      self.m[k] = np.dot(self.C[k],np.dot(self._eS[k],mm[k]))
      self.V[k] = self._D + np.dot((self.z[:,k] * dobs.T), dobs) \
          + zsum[k] * self.C[k]
      #for n in xrange(nobs):
      #  dobs = obs[n] - self.m[k]
      #  self.V[k] += self.z[n,k] * np.outer(dobs,dobs)

  def _VBE(self,obs,use_ext=True):
    self._calcSufficientStatistic(obs)
    self.z = self._evaluateHiddenState(obs,use_ext)

  def _VBM(self,obs):
    self._updatePosteriorParameters(obs)

  def _VBLowerBound(self,obs,use_ext=True):
    # negative of variational lower bound

    nobs,ndim = obs.shape
    nmix = self._nstates
    lnz = np.log(self.z)
    lnpi = np.log(self.pi)
    zsum = self.z.sum(0)
    
    # <lnP(D|z,theta)>
    lnPD = 0.0
    if use_ext and ext_imported:
      for k in xrange(nmix):
        lnPD = zsum[k]*(self._elnS[k] - np.trace(np.dot(self._eS[k],self.C[k])))
      lnPD += _lnPD_C(self.z,obs,nobs,nmix,self.m,self._eS)
    else:
      # very slow! neew Fortran or C codes
      for k in xrange(nmix):
        dobs = obs - self.m[k]
        lnPD = zsum[k]*(self._elnS[k] - np.trace(np.dot(self._eS[k],\
            self.C[k]))) + np.dot(self.z[:,k],np.diag(np.dot(dobs,\
            np.dot(self._eS[k],dobs.T))))
    lnPD *= 0.5

    # KL(q(z)||p(z))
    KLz = 0.0
    for k in xrange(nmix):
      KLz += np.dot(self.z[:,k],lnz[:,k] - lnpi[k])

    # KL(q(mu)||p(mu))
    KLmu = 0.0
    for k in xrange(nmix):
      KLmu = KLmu - np.log(det(self.C[k])) + self._beta \
          * (np.trace(self.C[k]) + np.dot(self.m[k],self.m[k]))
    KLmu *= 0.5

    # KL(q(S)||p(S))
    KLS = 0.0
    lnV = np.log([det(Vk) for Vk in self.V])
    KLS = 0.5 * (np.dot(self.nu,lnV) + np.dot(zsum,self._elnS) \
        + np.trace(np.dot(self._D,self._eS.sum(0)))) \
        - gammaln((np.arange(1,ndim+1) + self.nu[np.newaxis].T)/2).sum()

    return lnPD - KLz - KLmu - KLS

  def _VBFreeEnergy(self,obs,use_ext=True):
    return - self._VBLowerBound(obs,use_ext)

  def fit(self,obs,niter=200,eps=1e-4,use_ext=True,use_emgmm=False,debug=False):
    self._init_params(readObs(obs),use_emgmm)
    F_old = 1.0e50
    self._VBE(obs,use_ext)
    for i in range(niter):
      self._VBM(obs)
      self._VBE(obs,use_ext)
      if debug:
        self._posterior_parameters()
      F_new = self._VBFreeEnergy(obs,use_ext)
      dF = F_new - F_old
      print "%8dth iter, Free Energy = %10.4e, dF = %10.4e" %(i,F_new,dF)
      if abs(dF) < eps and dF < 0:
        print dF, " < ", eps, "Converged"
        break
      F_old = F_new
    return self

  def decode(self,obs):
    z = self._evaluateHiddenState(readObs(obs))
    codes = z.argmax(1)
    clust = [[] for i in range(z.shape[1])]
    for (o,c) in (obs,codes):
      clust[c].append(obs)
    for cl in clust:
      cl = np.array(cl)
    return codes,clust

def test1(nmix,niter=10000):
  from scikits.learn.mixture import GMM
  Y = testData2(2000)
  #from scikits.learn import datasets
  #iris = datasets.load_iris()
  #Y = iris.data
  #label = iris.target
  model = VBGMM(nmix)
  model.fit(Y,niter)
  C = np.array([[0., -0.7], [3.5, .7]])
  print "original covarience","\n",np.dot(C.T,C)
  for k in xrange(nmix):
    if model.pi[k] > 1.0e-4:
      print "\n\n%dth cluster" % k
      print "mixture coefficient","\n", model.pi[k],"\n"
      print "mean","\n", model.m[k],"\n"
      print "covarience","\n",model.V[k] / model.nu[k],"\n"
  model2 = GMM(2,"full")
  model2.fit(Y)
  for i in range(2):
    print model2._get_weights()[i]
    print model2._means[i]
    print model2._covars[i]

if __name__ == "__main__":
  from sys import argv
  nmix = int(argv[1])
  test1(nmix)
