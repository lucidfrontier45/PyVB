#!/usr/bin/python

import numpy as np
from numpy.random import randn, dirichlet
from scipy.linalg import det, inv
from scipy.cluster import vq
from util import logsum, log_like_Gauss
from sampling import testData

class EMGMM:
    """
    Gaussian Mixture Model with Expectation-Maximization (EM) Algorithm.
    """ 
    def __init__(self,nmix=10):
        # maximum number of the hidden clusters
        self._nstates = nmix
    
    def _init_params(self,obs):
        nmix = self._nstates
        nobs, ndim = obs.shape
        self._init_prior(obs)
        self._init_posterior(obs)
        
        self._C = np.empty((self._nstates,nmix,ndim,ndim))

    def _init_prior(self,obs):
        pass
    
    def _init_posterior(self,obs):
        nmix = self._nstates
        nobs, ndim = obs.shape
        self.z = np.ones((nobs,nmix)) / float(nmix)
        self.pi = np.ones(nmix) / float(nmix)
        self.m, temp = vq.kmeans2(obs,nmix)
        self.cv = np.tile(np.cov(obs.T),(nmix,1,1))
          
    def showModel(self,show_m=False,show_cv=False,min_pi=0.01):
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
            print "\n%dth component, pi = %8.3g" % (k,params[k][0])
            print "cluster id =", params[k][1]
            if show_m:
                print "mu =",params[k][2]
            if show_cv:
                print "tau =",params[k][3]
        return relavent_clusters

    def _log_like_f(self,obs):
        lnf = np.log(self.pi)[np.newaxis,:] \
            + log_like_Gauss(obs,self.m,self.cv) 
        return lnf

    def eval_hidden_states(self,obs):
        lnf = self._log_like_f(obs)
        lnP = logsum(lnf,1)
        z = lnf - lnP[:,np.newaxis]
        return z,lnP.sum()
    
    def fit(self,obs,niter=1000,eps=1.0e-4,ifreq=10,init=True):
        if init:
            self._init_params(obs)
        
        F = 1.0e50
        
        for i in xrange(niter):
            F_new = self._E_step(obs)
            dF = F_new - F
            if abs(dF) < eps :
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" \
                %(i,F_new,dF)
                print "%12.6e < %12.6e Converged" %(dF, eps)
                break
            if i % ifreq == 0 and dF < 0.0:
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" \
                %(i,F_new,dF)
            elif dF > 0.0:
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e warning" \
              %(i,F_new,dF)
                  
            F = F_new
            self._M_step(obs)
    
    def _E_step(self,obs):
        self.z, lnP = self.eval_hidden_states(obs)      
        return -lnP
        
    def _M_step(self,obs):
        self._calculate_sufficient_statistics(obs)
        self._update_parameters()
        
    def _calculate_sufficient_statistics(self,obs):
        nmix = self._nstates        
        self._N = self.z.sum(0)
        self._xbar = np.dot(self.z.T,obs) / self._N[:,np.newaxis]
        for k in xrange(nmix):
            dobs = obs - self._xbar[k]
            self._C[k] = np.dot((self.z[:,k] * dobs.T), dobs)
        
    def _update_parameters(self):
        nmix = self._nstates
        self.pi = self._N / self._N.sum()
        self.m = np.array(self._xbar)
        self.cv = self._C / self._N[:,np.newaxis,np.newaxis]
        
    
def test1(n=1000):
    X = testData(n)
    model = EMGMM(5)
    model.fit(X)
    model.showModel()
    
if __name__ == "__main__":
    from sys import argv
    test1(int(argv[1]))