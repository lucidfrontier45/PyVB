#!/usr/bin/python

import numpy as np
from numpy.random import randn,dirichlet
from scipy.linalg import det, inv
from scipy.cluster import vq
from scipy.special import psi,gammaln
from core import *

class EMGMM:
  """
  Gaussian Mixture Model with Expectation-Maximization (EM) Algorithm.
  """
  def __init__(self,nmix=10,u=0.5,m=0.0,beta=1,nu=1,s=10.0):
    # maximum number of the hidden clusters
    self._nstates = nmix 

  def _init_params(self,obs,adjust_prior=True,scale=10.0):
    self._set_posterior(obs)

  def _set_posterior(self,obs):
    """
    Initialize posterior parameters before running iterative fitting
    """
    nobs, D = obs.shape
    nmix = self._nstates
    # hidden states
    self.z = dirichlet(np.tile(1.0/nmix,nmix),nobs)
    # mixing coefficients
    self.pi = self.z.sum(0) / float(nobs)
    # posterior mean vector
    self.m, temp = vq.kmeans2(obs,nmix)
    # posterior covariance
    self.cv = np.tile(np.identity(D),(nmix,1,1))

    # aux
    # unnormalized sample covariances
    self.C = np.array(self.cv)

  def _log_like_f(self,obs):
    """
    mean log-likelihood function
    <lnP(X,Z|theta)>_Q(theta) = <lnP(Z|pi)>_Q(pi) + <lnP(X|Z,mu,Sigma>_Q(mu,Sigma)
    """
    nobs, ndim = obs.shape
    nmix = self._nstates
    lnr = np.log(self.pi)[np.newaxis,:] \
      + lmvnpdf(obs,self.m,self.cv,"full") 
    return lnr

  def _VBE(self,obs):
    """
    VB-E step
    Calculate expectations over variational posteriors Q(Z,theta) and obtain
    Q(Z)
    """
    # calc Q(Z)
    self.z,lnP = self._evaluateHiddenState(obs)
    return lnP

  def _evaluateHiddenState(self,obs):
    """
    Calc Q(Z) = (1/C) * exp<lnP(X,Z|theta)>_Q(theta)
    input
      obs [ndarray, shape (N,D)] : observed data
    output
      z [ndarray, shape (N,nmix)] : posterior probabiliry of hidden states,
        z[n,k] = Q(Zn=k)
      lnP [float] : <lnP(X,Z|theta)>_Q(Z,theta) - <lnQ(Z)>_Q(Z)
    """
    lnr = self._log_like_f(obs) # <lnP(X,Z|theta)>_Q(theta)
    norm_const = logsum(lnr,1) # log normalizr constant for <lnP(Xn,Zn|theta)>
    z = np.exp(lnr - norm_const[:,np.newaxis]) # convert <lnP(X,Z|theta> 
    lnP = norm_const.sum()
    return z, lnP

  def _VBM(self,obs):
    """
    VB-M step calculates sufficient statistics and use them to update parameters of posterior distribution
    """
    self._calcSufficientStatistic(obs)
    self._updatePosteriorParameters(obs)

  def _calcSufficientStatistic(self,obs):
    """
    Calculate sufficient Statistics
    """
    nmix = self._nstates
    # sum_n Q(Zn=k)
    self.N = self.z.sum(0)
    # sum_n Q(Zn=k)Xn / Nk
    self.xbar = np.dot(self.z.T,obs) / self.N[:,np.newaxis]
    for k in xrange(nmix):
      dobs = obs - self.xbar[k]
      self.C[k] = np.dot((self.z[:,k]*dobs.T),dobs)

  def _updatePosteriorParameters(self,obs):
    """
    Update parameters of variational posterior distribution by precomputed
    sufficient statistics
    """
    nobs, D = obs.shape
    nmix = self._nstates
    # parameter for mixing coefficients
    self.pi = self.N / float(nobs)
    # parameters for mean 
    self.m = np.array(self.xbar)
    # parameters for covariance 
    self.cv = self.C / self.N[:,np.newaxis,np.newaxis]

  def eval(self,obs):
    """
    Calculate variational free energy 
    """
    z,lnP = self._evaluateHiddenState(obs)
    return -lnP

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

    # main loop
    for i in range(niter):
      # VB-E step
      lnP = self._VBE(obs)
      # check for convergence
      F_new = -lnP 
      dF = F_new - F_old
      if abs(dF) < eps :
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F_new,dF)
        print "%12.6e < %12.6e Converged" %(dF, eps)
        break
      if i % ifreq == 0 and dF < 0.0:
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F_new,dF)
      elif dF > 0.0:
        print "%8dth iter, Free Energy = %12.6e, dF = %12.6e warning" \
              %(i,F_new,dF)
      F_old = F_new
      # VB-Mstep
      self._VBM(obs)
    if plot:
      self.plot2d(obs)
    return self

  def showModel(self,show_m=False,show_s=False,min_pi=0.01):
    """
    Obtain model parameters for relavent clusters
    """
    nmix = self._nstates
    params = sorted(zip(self.pi,range(nmix),self.m,self.cv),\
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
    pi = -np.sort(-self.pi)[:len(params)]
    pi = pi / pi.sum()
    y = np.array([GaussianPDF(x,p[1],p[2]) * pi[k] \
        for k,p in enumerate(params)])
    return y

  def decode(self,obs,min_pi=0.01):
    z,lnP = self._evaluateHiddenState(obs)
    codes = z.argmax(1)
    params = self.showModel(min_pi)
    clust_pos = []
    for p in params:
      k = p[1]
      clust_pos.append(codes==p[1])
    return clust_pos

  def plot1d(self,obs,d1=0,min_pi=0.01,clust_pos=None):
    symbs = ".hd^x+"
    l = np.arange(len(obs))
    if clust_pos == None:
      clust_pos = self.decode(obs,min_pi)
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

  def plot2d(self,obs,d1=0,d2=1,min_pi=0.01,clust_pos=None):
    symbs = ".hd^x+"
    if clust_pos == None:
      clust_pos = self.decode(obs,min_pi)
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
  
  def makeTransMat(self,obs,norm=True,min_degree=10):
    """
    Make transition matrix from posterior Q(Z)
    """

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

  def AIC(self,obs):
    """
    Akaike Information Criterion
    """
    k = complexity_GMM(self._nstates,len(self.m[0]))
    return self.eval(obs) + k

  def BIC(self,obs):
    """
    Bayes Information Criterion
    """
    k = complexity_GMM(self._nstates,len(self.m[0]))
    return self.eval(obs) + k * np.log(len(obs))

def test1(nmix,niter=10000):
  Y = testData(1000)
  #Y = cnormalize(Y)
  model = EMGMM(nmix)
  model.fit(Y,niter,ifreq=10,plot=False)
  model.showModel()
  print model.AIC(Y), model.BIC(Y)


if __name__ == "__main__":
  from sys import argv
  nmix = int(argv[1])
  test1(nmix)
