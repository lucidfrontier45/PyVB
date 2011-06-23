#!/usr/bin/python

import numpy as np
from numpy.random import randn,dirichlet
from scipy.linalg import det, inv
from scipy.cluster import vq
from scipy.special import psi,gammaln
from core import *
from _core import _logsum,_logsum2d
try:
  from _vbgmm import _evaluateHiddenState_C
  ext_imported = True
except:
  ext_imported = False
  print "warning, Cython extension module was not found"
  print "computation can be slower"

class VBGMM:
  def __init__(self,nmix=10,u=0.5,m=0.0,beta=1,nu=1,s=10.0):
    self._nstates = nmix
    self._u0 = np.ones(nmix) * u  # Jeffrey prior if u0 = 0.5
    self._m0 = m
    self._beta0 = beta
    self._nu0 = nu
    self._s0 = s

  def _init_params(self,obs,adjust_prior=True,scale=10.0,use_emgmm=False):
    dim = obs.shape[1]
    if self._nu0 < dim:
      self._nu0 += dim
    if adjust_prior:
      self._adjust_prior(obs,scale)
    else :
      self._m0 = np.zeros(dim)
      self._s0 = np.identity(dim) * self._s0 
    self._set_posterior(obs,use_emgmm)

  def _adjust_prior(self,obs,scale=10.0):
    self._m0 = np.mean(obs,0)
    self._s0 = np.cov(obs.T) * scale

  def _set_posterior(self,obs,use_emgmm=False):
    nobs = len(obs)
    nmix = self._nstates
    # hidden states
    self.z = dirichlet(np.tile(1.0/nmix,nmix),nobs)
    # mixing coefficients
    self.u = np.array(self._u0)
    # posterior mean vector
    self.m, temp = vq.kmeans2(obs,nmix)
    self.beta = np.tile(self._beta0,nmix)
    # posterior degree of freedom
    self.nu = np.tile(float(nobs)/nmix,nmix)
    # posterior precision
    self.s = np.tile(np.array(self._s0),(nmix,1,1))

    # aux
    self._et = np.array(self.s)
    self.C = np.array(self.s)

  def _VBE(self,obs):
    nmix = self._nstates
    nobs, ndim = obs.shape
    self._epi = self.u / self.u.sum() # <pi_k>
    self._elnpi = psi(self.u) - psi(self.u.sum()) # <ln(pi_k)>
    self._et = self.s / self.nu[:,np.newaxis,np.newaxis] # <tau_k>
    self.z,lnP = self._evaluateHiddenState(obs)
    return lnP

  def _evaluateHiddenState(self,obs):
    nobs, ndim = obs.shape
    nmix = self._nstates
    z = lmvnpdf2(obs,self.m,self._et,self.nu,self.beta) \
        + self._elnpi[np.newaxis,:]
    zsum = logsum(z,1)
    #zsum = _logsum2d(nobs,ndim,z)
    z = np.exp(z - zsum[:,np.newaxis])
    #z = np.exp(z - z.max(1)[np.newaxis].T)
    #z = normalize(z,1)
    return z,zsum.sum()

  def _VBM(self,obs):
    self._calcSufficientStatistic(obs)
    self._updatePosteriorParameters(obs)

  def _calcSufficientStatistic(self,obs):
    nmix = self._nstates
    self.N = self.z.sum(0)
    self.xbar = np.dot(self.z.T,obs) 
    for k in xrange(nmix):
      self.xbar[k] /= self.N[k]
      dobs = obs - self.xbar[k]
      self.C[k] = np.dot((self.z[:,k]*dobs.T),dobs)

  def _updatePosteriorParameters(self,obs):
    nmix = self._nstates
    self.u = self._u0 + self.N
    self.beta = self.N + self._beta0
    self.nu = self._nu0 + self.N
    self.s = self._s0 + self.C
    for k in xrange(nmix):
      self.m[k] = (self._beta0 * self._m0 + self.N[k] * self.xbar[k])\
          / self.beta[k]
      dx = self.xbar[k] - self._m0
      self.s[k] += (self._beta0 * self.N[k] / self.beta[k]) * np.dot(dx.T, dx)

  def _KL_div(self):
    nmix = self._nstates
    KL = KL_Dirichlet(self.u,self._u0)
    for k in xrange(nmix):
      KL += KL_GaussWishart(self.nu[k],self.s[k],self.beta[k],self.m[k],\
          self._nu0,self._s0,self._beta0,self._m0)
    return KL

  def eval(self,obs):
      lnP = self._VBE(obs)
      F = -lnP + self._KL_div()
      return F

  def fit(self, obs, niter=10000, eps=1e-4, ifreq=50,\
        init=True, plot=True):
    if init : self._init_params(obs)
    F_old = 1.0e50
    for i in range(niter):
      lnP = self._VBE(obs)
      F_new = -lnP + self._KL_div()
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
      F_old = F_new
      self._VBM(obs)
    if plot:
      self.plot2d(obs)
    return self

  def showModel(self,show_m=False,show_s=False,min_pi=0.01):
    nmix = self._nstates
    params = sorted(zip(self._epi,range(nmix),self.m,self._et),\
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
      if show_s:
        print "tau =",params[k][3]
    return relavent_clusters

  def pdf(self,x,min_pi=0.01):
    params = self.showModel(min_pi)
    pi = -np.sort(-self._epi)[:len(params)]
    pi = pi / pi.sum()
    y = np.array([GaussianPDF(x,p[1],p[2]) * pi[k] \
        for k,p in enumerate(params)])
    return y

  def decode(self,obs):
    z,lnP = self._evaluateHiddenState(obs)
    codes = z.argmax(1)
    params = self.showModel()
    clust_pos = []
    for p in params:
      k = p[1]
      clust_pos.append(codes==p[1])
    return clust_pos

  def plot1d(self,obs,d1=0,clust_pos=None):
    symbs = ".hd^x+"
    l = np.arange(len(obs))
    if clust_pos == None:
      clust_pos = self.decode(obs)
    try :
      import matplotlib.pyplot as plt
    except ImportError :
      print "cannot import pyplot"
      return
    for k,pos in enumerate(clust_pos):
      symb = symbs[k / 7]
      plt.plot(l[pos],obs[pos,d1],symb,label="%3dth cluster"%k)
    plt.legend(loc=0)
    plt.show()

  def plot2d(self,obs,d1=0,d2=1,clust_pos=None):
    symbs = ".hd^x+"
    if clust_pos == None:
      clust_pos = self.decode(obs)
    try :
      import matplotlib.pyplot as plt
    except ImportError :
      print "cannot import pyplot"
      return
    for k,pos in enumerate(clust_pos):
      symb = symbs[k / 7]
      plt.plot(obs[pos,d1],obs[pos,d2],symb,label="%3dth cluster"%k)
    plt.legend(loc=0)
    plt.show()
  
  def plot3d(self,obs,d1=0,d2=1,d3=2,clust_pos=None):
    # unfinished ..
    if clust_pos == None:
      clust_pos = self.decode(obs)
    try :
      import matplotlib.pyplot as plt
      from mpl_toolkits.mplot3d import Axes3D
    except ImportError :
      print "cannot import pyplot"
      return
    fig = plt.figure()
    axis = Axes3D(fig)
    for k,pos in enumerate(clust_pos):
      axis.scatter(obs[pos,d1],obs[pos,d2],obs[pos,d3],label="%3dth cluster"%k)
    plt.legend(loc=0)
    fig.show()

  def makeTransMat(self,obs,norm=True,min_degree=10):
    # MT[i,j] = N(x_{t+1}=j|x_t=i)
    z,lnP = self._evaluateHiddenState(obs)
    codes = z.argmax(1)
    dim = self._nstates
    MT = np.zeros((dim,dim))
    for t in xrange(1,len(z)-1):
      #MT[codes[t],codes[t+1]] += 1
      MT += np.outer(z[t-1],z[t])
      #for i in xrange(dim):
      #  for j in xrange(dim):
      #    MT[i,j] += z[t-1,i] * z[t,j]
    for i in xrange(len(MT)):
      for j in xrange(len(MT)):
        if MT[i,j] < min_degree:
          MT[i,j] = 0.0

    # extract relavent cluster
    params = self.showModel()
    cl = [p[1] for p in params]
    MT = np.array([mt[cl] for mt in MT[cl]])

    if norm:
      # MT[i,j] = P(x_{t+1}=j|x_t=i)
      MT = normalize(MT,1)
    
    return MT

def test1(nmix,niter=10000):
  Y = testData(1000)
  #Y = cnormalize(Y)
  model = VBGMM(nmix)
  model.fit(Y,niter,plot=True)

if __name__ == "__main__":
  from sys import argv
  nmix = int(argv[1])
  test1(nmix)
