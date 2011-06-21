#!/usr/bin/python

import numpy as np
from numpy.random import randn,dirichlet
from scipy.linalg import det, inv
from scipy.cluster import vq
from scipy.special import psi,gammaln
from emgmm import EMGMM
from moments import *
from sampling import testData

class VBGMM(EMGMM):
  """
  Gaussian Mixture Model with Variational Bayesian Learning
  This class is mostly based on the EMGMM class.
  """
  def __init__(self,nmix=10,u=0.5,m=0.0,beta=1,nu=1,s=10.0):
    # maximum number of the hidden clusters
    self._nstates = nmix
    # hyper parameter for Dirichlet prior of mixing coefficients
    self._u0 = np.ones(nmix) * u
    # hyperparameters for prior of mean
    self._m0 = m
    self._beta0 = beta
    # hyper parameters for prior of covariance
    self._nu0 = nu
    self._s0 = s

  def _init_params(self,obs,adjust_prior=True,scale=10.0):
    """
    Initialize prior and posterior parameters
    """
    dim = obs.shape[1]
    # make sure the prior dof parameter of Wishart is valid
    if self._nu0 < dim:
      self._nu0 += dim
    if adjust_prior:
      self._adjust_prior(obs,scale)
    else :
      self._m0 = np.zeros(dim)
      self._s0 = np.identity(dim) * self._s0
    self._set_posterior(obs)

  def _adjust_prior(self,obs,scale=10.0):
    """
    Adjust prior parameters according to observed data
    """
    self._m0 = np.mean(obs,0)
    self._s0 = np.cov(obs.T) * scale

  def _set_posterior(self,obs):
    """
    Initialize posterior parameters before running iterative fitting
    """
    nobs = len(obs)
    nmix = self._nstates
    # hidden states
    self.z = dirichlet(np.tile(1.0/nmix,nmix),nobs)
    # mixing coefficients
    self._u = np.array(self._u0)
    # posterior mean vector
    self._m, temp = vq.kmeans2(obs,nmix)
    self._beta = np.tile(self._beta0,nmix)
    # posterior degree of freedom
    self._nu = np.tile(float(nobs)/nmix,nmix)
    # posterior covariance
    self._s = np.tile(np.array(self._s0),(nmix,1,1))

    # aux
    # unnormalized sample covariances
    self.C = np.array(self._s)

  def _log_like_f(self,obs):
    """
    mean log-likelihood function
    <lnP(X,Z|theta)>_Q(theta) = <lnP(Z|pi)>_Q(pi) + <lnP(X|Z,mu,Sigma>_Q(mu,Sigma)
    """
    nobs, ndim = obs.shape
    nmix = self._nstates
    z = self.elnpi[np.newaxis,:] \
      + lmvnpdf2(obs,self._m,self.ecv,self._nu,self._beta)
    return z

  def _VBE(self,obs):
    """
    VB-E step
    Calculate expectations over variational posteriors Q(Z,theta) and obtain
    Q(Z)
    """
    nmix = self._nstates
    nobs, ndim = obs.shape

    # calc expectations
    self.epi = E_pi_Dirichlet(self._u) # <pi_k>
    self.elnpi = E_lnpi_Dirichlet(self._u) # <ln(pi_k)>
    self.ecv = [E_V_Wishart(self._nu[k],self._s[k]) for k in xrange(nmix)] # <Cov_k>
    # calc Q(Z)
    self.z,lnP = self._evaluateHiddenState(obs)
    return lnP

  def _VBM(self,obs):
    """
    VB-M step calculates sufficient statistics and use them to update parameters of posterior distribution
    """
    self._calcSufficientStatistic(obs)
    self._updatePosteriorParameters(obs)

  def _updatePosteriorParameters(self,obs):
    """
    Update parameters of variational posterior distribution by precomputed
    sufficient statistics
    """
    nmix = self._nstates
    # parameter for mixing coefficients
    self._u = self._u0 + self.N
    # parameters for mean and covariance matrix which obey Gauss-Wishart dist
    self._nu = self._nu0 + self.N
    self._s = self._s0 + self.C
    self._beta = self._beta0 + self.N
    for k in xrange(nmix):
      self._m[k] = (self._beta0 * self._m0 + self.N[k] * self.xbar[k])\
          / self._beta[k]
      dx = self.xbar[k] - self._m0
      self._s[k] += (self._beta0 * self.N[k] / self._beta[k]) * np.dot(dx.T, dx)

  def _KL_div(self):
    """
    Calculate KL-divergence of parameter distribution KL[Q(theta)||P(theta)]
    """
    nmix = self._nstates
    # KL for mixing coefficients
    KL = KL_Dirichlet(self._u,self._u0)
    # KL for mean and covariance matrix
    for k in xrange(nmix):
      KL += KL_GaussWishart(self._nu[k],self._s[k],self._beta[k],self._m[k],\
          self._nu0,self._s0,self._beta0,self._m0)
    if KL < 0.0 :
      raise ValueError, "KL must be larger than 0"
    return KL

  #def compareCluster(self,i,j,):
  #  KL1 = KL_GaussWishart(self.nu[i],self.s[i],self.beta[i],self.m[i],\
  #                        self.nu[j],self.s[j],self.beta[j],self.m[j])
  #  KL2 = KL_GaussWishart(self.nu[j],self.s[j],self.beta[j],self.m[j],\
  #                        self.nu[i],self.s[i],self.beta[i],self.m[i])
  #  return 0.5 * (KL1 + KL2)

  #def mergeCluster(self,i,j,obs,copy=False,update=True):
  #  if copy:
  #    old_z = np.array(self.z)

  #  self.z[:,i] += self.z[:,j]
  #  self.z[:,j] = 1.0e-15

  #  if update:
  #    self._VBM(obs)
  #    self._epi = self.u / self.u.sum() # <pi_k>
  #    self._elnpi = psi(self.u) - psi(self.u.sum()) # <ln(pi_k)>
  #    self._et = self.s / self.nu[:,np.newaxis,np.newaxis] # <tau_k>
  #  if copy:
  #    return old_z

  def eval(self,obs):
    """
    Calculate variational free energy
    """
    z,lnP = self._evaluateHiddenState(obs)
    F = -lnP + self._KL_div()
    return F

  def fit(self, obs, niter=10000, eps=1e-4, ifreq=50,\
        init=True, plot=True):
    """
    Fit model parameters via VB-EM algorithm
    input
      obs [ndarray, shape(N,D)] : observed data
      niter [int] : maximum number of iteration cyles
      eps [float] : convergence threshold
      ifreq [int] : frequency of printing fitting process
      init [bool] : flag for initialization
      plot [bool] : flag for plotting the result
    """
    if init : self._init_params(obs)
    F_old = 1.0e50


    KL_old = self._KL_div()
    lnP_old = -1.0e50

    # main loop
    for i in range(niter):
      # VB-E step
      lnP = self._VBE(obs)
      # check for convergence
      KL = self._KL_div()

      dlnP = lnP -lnP_old
      dKL = KL - KL_old
      F_new = -lnP + KL
      dF = F_new - F_old
      dF2 = -dlnP + dKL
      if abs(dF2) < eps :
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F_new,dF2)
        print "%12.6e < %12.6e Converged" %(dF, eps)
        break
      if i % ifreq == 0 and dF2 < 0.0:
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F_new,dF2)
      elif dF > 0.0:
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e warning" \
              %(i,F_new,dF2)
      F_old = F_new
      # VB-Mstep
      self._VBM(obs)
      KL_old = KL
      lnP_old = lnP
    if plot:
      self.plot2d(obs)
    return self

  def showModel(self,show_m=False,show_cv=False,min_pi=0.01):
    """
    Obtain model parameters for relavent clusters
    """
    nmix = self._nstates
    params = sorted(zip(self.epi,range(nmix),self._m,self.ecv),\
        key=lambda x:x[0],reverse=True)
    relavent_clusters = []
    for k in xrange(nmix):
      if params[k][0] < min_pi:
        break
      relavent_clusters.append(params[k])
      print "\n%dth component, pi = %8.3g" \
          % (k,params[k][0])
      print "cluster id =", params[k][1]
      if show_m:
        print "mu =",params[k][2]
      if show_cv:
        print "tau =",params[k][3]
    return relavent_clusters

  def pdf(self,x,min_pi=0.01):
    params = self.showModel(min_pi)
    pi = -np.sort(-self.epi)[:len(params)]
    pi = pi / pi.sum()
    y = np.array([GaussianPDF(x,p[1],p[2]) * pi[k] \
        for k,p in enumerate(params)])
    return y

def test1(nmix,niter=10000):
  Y = testData(1000)
  #Y = cnormalize(Y)
  model = VBGMM(nmix)
  model.fit(Y,niter,ifreq=10,plot=False)
  model.showModel(True,True)

if __name__ == "__main__":
  from sys import argv
  nmix = int(argv[1])
  test1(nmix)
