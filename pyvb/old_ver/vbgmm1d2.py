#!/usr/bin/python

import numpy as np
from numpy.random import randn,dirichlet
from scipy.linalg import det, inv
from scipy.cluster import vq
from scipy.special import psi,gammaln
from core import *
try:
  from _vbgmm1d import _evaluateHiddenState_C, _lnPD_C
  ext_imported = True
except:
  ext_imported = False
  #print "warning, Cython extension module was not found"
  #print "computation can be slower"

def testData1(n=100):
  X = np.r_[randn(n*2)]
  return X

def testData2(n=100):
  X = np.r_[randn(n*2) / 0.3 , randn(n) + 10.0, randn(n*3) / 0.5 - 10.0, randn(3*n) / 0.1]
  return X

def GaussianPDF(x,mu,s):
  return np.exp(-((x - mu)**2)*s*0.5)*np.sqrt(s/(2.0*np.pi))

def lnZ_Wishart(nu,V):
  # log normalization constant of 1D Wishart 
  lnZ = 0.5 * nu * np.log(2.0*V) + gammaln(nu * 0.5)
  return lnZ


class VBGMM1D:
  def __init__(self,nmix=10,u=0.5,m=0.0,beta=1,nu=1,s=0.01):
    self._nstates = nmix
    self._u0 = u  # Jeffrey prior if u0 = 0.5
    self._m0 = m
    self._beta0 = beta
    self._nu0 = nu
    self._s0 = s

  def _init_params(self,obs,adjust_prior=True,scale=0.1,use_emgmm=False):
    if adjust_prior:
      self._adjust_prior(obs,scale)
    self._set_posterior(obs,use_emgmm)

  def _adjust_prior(self,obs,scale=0.1):
    self._m0 = np.mean(obs)
    self._s0 = 1.0 / (np.std(obs) * self._nu0) * scale

  def _set_posterior(self,obs,use_emgmm=False):
    nobs = len(obs)
    nmix = self._nstates
    # hidden states
    self.z = dirichlet(np.tile(1.0/nmix,nmix),nobs)
    # mixing coefficients
    self.u = np.tile(self._u0,nmix)
    # posterior mean vector
    self.m, temp = vq.kmeans2(obs,nmix)
    self.beta = np.tile(self._beta0,nmix)
    # posterior degree of freedom
    self.nu = np.tile(float(nobs)/nmix,nmix)
    # posterior precision
    self.s = np.tile(self._s0,nmix)

  def _VBE(self,obs,use_ext=True):
    self._epi = self.u / self.u.sum() # <pi_k>
    self._elnpi = psi(self.u) - psi(self.u.sum()) # <ln(pi_k)>
    self._et = self.s * self.nu # <tau_k>
    self._elnt = psi(self.nu*0.5) + np.log(2.0*self.s) # <ln(t_k)>
    self.z = self._evaluateHiddenState(obs,use_ext)

  def _evaluateHiddenState(self,obs,use_ext=True):
    nobs = len(obs)
    nmix = self._nstates
    ln2pi = np.log(2.0 * np.pi)
    z = np.tile(self._elnpi + 0.5 * self._elnt - 0.5 * ln2pi ,(nobs,1))
    if use_ext and ext_imported :
      pass
    else :
      for k in xrange(nmix):
        # very slow! need Fortran or C codes
        dobs = obs - self.m[k]
        z[:,k] -= 0.5 * (1.0/self.beta[k] + self.nu[k]*self.s[k]*(dobs**2))
    z = np.exp(z - z.max(1)[np.newaxis].T)
    z = normalize(z,1)
    return z

  def _VBM(self,obs):
    self._calcSufficientStatistic(obs)
    self._updatePosteriorParameters(obs)

  def _calcSufficientStatistic(self,obs):
    self.N = self.z.sum(0)
    self.xbar = np.dot(obs,self.z) / self.N
    self.C = np.diag(np.dot(((obs - self.xbar[np.newaxis].T)**2),self.z))

  def _updatePosteriorParameters(self,obs):
    self.u = self._u0 + self.N
    self.beta = self.N + self._beta0
    self.m = (self._beta0 * self._m0 + self.N * self.xbar) / self.beta
    self.nu = self._nu0 + self.N
    self.s = 1.0 / (1.0/self._s0 + self.C + (self._beta0 *self.N / self.beta) \
                    * (self.xbar - self._m0)**2)

  def _VBLowerBound1(self,obs,use_ext=True):
    # variational lower bound
    nmix = self._nstates

    self.N = self.z.sum(0) # need to be updated !!

    # <lnp(X|Z,theta)>
    # very slow! neew Fortran or C codes
    lnpX = np.dot(self.N,(self._elnpi + 0.5 * self._elnt))
    for k in xrange(nmix):
      dobs = obs - self.m[k]
      lnpX -= self.N[k] * 1.0 / self.beta[k] + self.s[k] * self.nu[k] * \
          (dobs**2).sum()

    # H[q(z)] = -<lnq(z)>
    Hz = 0.0
    for k in xrange(nmix):
      Hz -= np.dot(self.z[:,k],np.log(self.z[:,k]))

    # KL[q(pi)||p(pi)]
    KLpi = ( - gammaln(self.u) + self.N * psi(self.u)).sum()

    # KL[q(mu,tau)||p(mu,tau)]
    #KLmt = ((self.N * self._elnt + self.nu * (self.s / self._s0 - 1.0 - \
    #    np.log(2.0 * self.s)) + np.log(self.beta) + self._beta0 / self.beta + \
    #    self.nu * self.s * self._beta0 * (self.m - self._m0)**2) * 0.5 - \
    #    gammaln(self.nu * 0.5)).sum()

    # Wishart part
    KLmt = (self.N * self._elnt + self.nu * (self.s / self._s0 - 1.0)).sum() \
        * 0.5 + nmix * lnZ_Wishart(self._nu0,self._s0)
    for k in xrange(nmix):
      KLmt  -= lnZ_Wishart(self.nu[k],self.s[k])
    
    # Conditional Gaussian part
    KLmt += 0.5 * (np.log(self.beta/self._beta0) + self._beta0/self.beta - 1 \
        + self._beta0 * self.nu * self.s * (self.m-self._m0)**2).sum()

    return lnpX + Hz - KLpi - KLmt

  def _VBLowerBound2(self,obs,use_ext=True):
    # variational lower bound

    nobs = len(obs)
    nmix = self._nstates

    self.N = self.z.sum(0) # need to be updated !!

    # H[q(z)] = -<lnq(z)>
    Hz = 0.0
    for k in xrange(nmix):
      Hz -= np.dot(self.z[:,k],np.log(self.z[:,k]))

    # KL[q(pi)||p(pi)]
    KLpi = (gammaln(nmix * self._u0 + nobs) - gammaln(nmix * self._u0)) \
          - gammaln(self.u).sum() + nmix * gammaln(self._u0)

    # KL[q(mu,tau)||p(mu,tau)]
    KLmt = (np.log(self.beta).sum() - nmix * self._beta0) * 0.5
    KLmt += lnZ_Wishart(self._nu0,self._s0) * nmix
    for k in xrange(nmix):
      KLmt  -= lnZ_Wishart(self.nu[k],self.s[k])

    #print "%12.5e %12.5e %12.5e"%(Hz,-KLpi,-KLmt)
    return Hz - KLpi - KLmt

  def _VBFreeEnergy(self,obs,use_ext=True):
    return - self._VBLowerBound2(obs,use_ext)

  def fit(self,obs,niter=200,eps=1e-4,ifreq=50,init=True,plot=False,\
      use_ext=False):
    if init : self._init_params(obs)
    F_old = 1.0e50
    for i in range(niter):
      old_u = np.copy(self.u)
      old_m = np.copy(self.m)
      old_s = np.copy(self.s)
      self._VBE(obs,use_ext)
      self._VBM(obs)
      F_new = self._VBFreeEnergy(obs,use_ext)
      dF = F_new - F_old
      if abs(dF) < eps :
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F_new,dF)
        print "%12.6e < %12.6e Converged" %(dF, eps)
        break
      if i % ifreq == 0:
        if dF < 0.0:
          print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F_new,dF)
        else :
          print "%8dth iter, Free Energy = %12.6e, dF = %12.6e warning" \
              %(i,F_new,dF)
      #conv_u = np.allclose(self.u,old_u)
      #conv_m = np.allclose(self.m,old_m)
      #conv_s = np.allclose(self.s,old_s)
      #if conv_u and conv_m and conv_s:
      #  break
      F_old = F_new
    if plot:
      self.plotPDF(obs)
    return self

  def showModel(self,min_pi=0.01):
    nmix = self._nstates
    params = sorted(zip(self._epi,self.m,self._et),reverse=True)
    relavent_clusters = []
    for k in xrange(nmix):
      if params[k][0] < min_pi:
        break
      relavent_clusters.append(params[k])
      print "%dth component, pi = %8.3g, mu = %8.3g, tau = %8.3g" \
          % (k+1,params[k][0],params[k][1],params[k][2])
    return relavent_clusters

  def pdf(self,x,min_pi=0.01):
    params = self.showModel(min_pi)
    pi = -np.sort(-self._epi)[:len(params)]
    pi = pi / pi.sum()
    y = np.array([GaussianPDF(x,p[1],p[2]) * pi[k] \
        for k,p in enumerate(params)])
    return y

  def plotPDF(self,obs,bins=100,min_pi=0.01):
    try :
      import matplotlib.pyplot as plt
    except ImportError :
      print "cannot import pyplot"
      return
    x = np.linspace(min(obs),max(obs),bins)
    y = self.pdf(x,min_pi)
    plt.hist(obs,bins,label="observed",normed=True)
    plt.plot(x,y.sum(0),label="sum",linewidth=8)
    for k,yy in enumerate(y) :
      plt.plot(x,yy,label="%dth cluster"%(k+1),linewidth=3)
    plt.legend(loc=0)
    plt.show()

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
  Y = testData2(2000)
  Y = cnormalize(Y)
  model = VBGMM1D(nmix)
  model.fit(Y,niter)
  model.showModel()
  #model.plotPDF(Y,bins=200)

if __name__ == "__main__":
  from sys import argv
  nmix = int(argv[1])
  test1(nmix)
