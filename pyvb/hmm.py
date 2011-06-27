#!/usr/bin/python

import numpy as np
from numpy.random import random,dirichlet
from scipy.cluster import vq
from util import logsum, log_like_Gauss, num_param_Gauss
from sampling import sample_gaussian

# import Fortran95 extension module
try:
    import _hmmf
    #import _hmmc
    ext_imported = True
except ImportError:
    print "extension module was not imported"
    ext_imported = False

class _BaseHMM():
    """
    Base HMM class.
    Any HMM class should be an inheritent of it.
    No instance of this class can be made.

    """

    def __init__(self,N):
        self._nstates = N # number of hidden states
        self._lnpi = np.log(np.tile(1.0/N,N)) # log initial probability
        self._lnA = np.log(dirichlet([1.0]*N,N)) # log transition probability

    def _log_like_f(obs):
        """
        Calculate log-likelihood of emissions
        """
        pass

    def _initialize_HMM(self,obs):
        """
        Do some initializations
        """
        nmix = self._nstates
        self.pi = np.empty(nmix)
        self.A = np.empty((nmix,nmix))

    def _allocate_temp(self,obs):
        """
        Allocate tempolary space for running forward-backward algorithm
        """
        T = len(obs)
        lnalpha = np.zeros((T,self._nstates)) #  log forward variable
        lnbeta = np.zeros((T,self._nstates)) # log backward variable
        lneta = np.zeros((T-1,self._nstates,self._nstates))
        return lnalpha, lnbeta, lneta

    def _forward(self,lnf,lnalpha,use_ext="F"):
        """
        Use forward algorith to calculate forward variables and loglikelihood
        input
          lnf [ndarray, shape (n,nmix)] : loglikelihood of emissions
          lnalpha [ndarray, shape (n,nmix)] : log forward variable
          use_ext [("F","C",None)] : flag to use extension
        output
          lnalpha [ndarray, shape (n.nmix)] : log forward variable
          lnP [float] : lnP(X|theta)
        """
        T = len(lnf)
        lnalpha *= 0.0
        if use_ext and ext_imported:
            if use_ext in ("c","C"):
                _hmmc._forward_C(T,self._nstates,self._lnpi,self._lnA,lnf,lnalpha)
            elif use_ext in ("f","F"):
                lnalpha = _hmmf.forward_f(self._lnpi,self._lnA,lnf)
            else :
                raise ValueError, "ext_use must be either 'C' or 'F'"
        else:
            lnalpha[0,:] = self._lnpi + lnf[0,:]
            for t in xrange(1,T):
                lnalpha[t,:] = logsum(lnalpha[t-1,:] + self._lnA.T,1) \
                    + lnf[t,:]
        return lnalpha,logsum(lnalpha[-1,:])

    def _backward(self,lnf,lnbeta,use_ext="F"):
        """
        Use backward algorith to calculate backward variables and loglikelihood
        input
          lnf [ndarray, shape (n,nmix)] : loglikelihood of emissions
          lnbeta [ndarray, shape (n,nmix)] : log backward variable
          use_ext [("F","C",None)] : flag to use extension
        output
          lnbeta [ndarray, shape (n,nmix)] : log backward variable
          lnP [float] : lnP(X|theta)
        """
        T = len(lnf)
        lnbeta *= 0.0
        if ext_imported:
            if use_ext in ("c","C"):
                _hmmc._backward_C(T,self._nstates,self._lnpi,self._lnA,lnf,lnbeta)
            elif use_ext in ("f","F"):
                lnbeta = _hmmf.backward_f(self._lnpi,self._lnA,lnf)
            else :
                raise ValueError, "ext_use must be either 'C' or 'F'"
        else:
            lnbeta[T-1,:] = 0.0
            for t in xrange(T-2,-1,-1):
                lnbeta[t,:] = logsum(self._lnA + lnf[t+1,:] + lnbeta[t+1,:],1)
        return lnbeta,logsum(lnbeta[0,:] + lnf[0,:] + self._lnpi)

    def eval_hidden_states(self,obs,use_ext="F"):
        """
        Performe one Estep.
        Then obtain variational free energy and posterior over hidden states
        """
        lnf = self._log_like_f(obs)
        lnalpha, lnbeta, lneta = self._allocate_temp(obs)
        lneta, lngamma, lnP = self._E_step(lnf,lnalpha,lnbeta,lneta,use_ext)
        return np.exp(lngamma), lnP

    def _complexity(self):
        """
        Count the number of parameter of HMM
        """
        comp = self._nstates * (1.0 + self._nstates)
        return comp

    def score(self,obs,mode,use_ext="F"):
        """
        score the model
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          mode [string] : one of 'ML', 'AIC' or 'BIC'
        output
          S [float] : score of the model
        """
        nobs, ndim = obs.shape
        z, lnP = self.eval_hidden_states(obs,use_ext="F")
        comp = self._complexity()
        if mode in ("AIC", "aic"):
            # use Akaike information criterion
            S = -lnP + comp
        if mode in ("BIC", "bic"):
            # use Bayesian information criterion
            S = -lnP + comp * np.log(nobs)
        else:
            # use negative likelihood
            S = -lnP
        return S

    def decode(self,obs,use_ext="F"):
        """
        Get the most probable cluster id
        """
        z,lnP = self.eval_hidden_states(obs)
        return z.argmax(1)
        
    def getRelaventCluster(self,eps=1.0e-2):
        """
        return parameters of relavent clusters
        """
        nmix = self._nstates
        ids = []
        sorted_ids = (-self.pi).argsort()
        for k in sorted_ids:
            if self.pi[k] > eps:
                ids.append(k)
        pi = self.pi[ids]
        A = np.array([AA[ids] for AA in self.A[ids]])
        return ids,pi,A

    def fit(self,obs,niter=1000,eps=1.0e-4,ifreq=10,init=True,use_ext="F"):
        """
        Fit the model parameters iteratively via EM algorithm
        """

        # performe initialization if needed
        if init:
              self._initialize_HMM(obs)
              old_F = 1.0e20

        # allocate temporary for Forward-Backward
        lnalpha, lnbeta, lneta = self._allocate_temp(obs)

        # main loop
        for i in xrange(niter):
            # E step
            lnf = self._log_like_f(obs)
            lneta, lngamma, lnP = \
            self._E_step(lnf,lnalpha,lnbeta,lneta,use_ext)
            # check for convergence
            F = -lnP
            dF = F - old_F
            if(abs(dF) < eps):
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e"\
                     %(i,F,dF)
                print "%12.6e < %12.6e Converged" %(dF, eps)
                break
            if i % ifreq == 0 and dF < 0.0:
                print "%6dth iter, F = %15.8e  df = %15.8e"%(i,F,dF)
            elif dF > 0.0:
                print "%6dth iter, F = %15.8e  df = %15.8e warning"%(i,F,dF)
            old_F = F
            # M step
            self._M_step(obs,lneta,lngamma,use_ext)
        
        self.pi = np.exp(self._lnpi)
        self.A = np.exp(self._lnA)
        return self

    def fit_multi(self,obss,niter=1000,eps=1.0e-4,ifreq=10,\
        init=True,use_ext="F"):
        """
        Performe EM step for multiple iid time-series
        """
        nobss = len(obss) # number of trajectories
        nobs = [len(obs) for obs in obss] # numbers of observations in all trajs
        i_max_obs = np.argmax(nobs)
        obs_flatten = np.vstack(obss) # flattened observations (sum(nobs)xdim)
        nmix = self._nstates

        # get posistion id for each traj
        # i.e. obss[i] = obs[pos_ids[i][0]:pos_ids[i][1]]
        pos_ids = []
        j = 0
        for i in xrange(nobss):
            pos_ids.append((j,j+nobs[i]))
            j += nobs[i]

        if init:
            self._initialize_HMM(obs_flatten)
            old_F = 1.0e20

        # allocate space for forward-backward
        lneta = []
        lngamma = []
        for nn in xrange(nobss):
            lneta.append(np.zeros((len(obss[nn])-1,nmix,nmix)))
            lngamma.append(np.zeros((len(obss[nn]),nmix)))
        lnalpha, lnbeta, lneta_temp = self._allocate_temp(obss[i_max_obs])

        for i in xrange(niter):
            lnP = 0.0
            lnf = self._log_like_f(obs_flatten)
            for nn in xrange(nobss):
                Ti,Tf = pos_ids[nn]
                e, g, p = self._E_step(lnf[Ti:Tf],lnalpha[:nobs[nn]],\
                    lnbeta[:nobs[nn]],lneta_temp[:nobs[nn]-1],use_ext)
                lneta[nn] = e[:]
                lngamma[nn] = g[:]
                lnP += p

            F = -lnP
            dF = F - old_F
            if(abs(dF) < eps):
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F,dF)
                print "%12.6e < %12.6e Converged" %(dF, eps)
                break
            if i % ifreq == 0 and dF < 0.0:
                print "%6dth iter, F = %15.8e  df = %15.8e"%(i,F,dF)
            elif dF > 0.0:
                print "%6dth iter, F = %15.8e  df = %15.8e warning"%(i,F,dF)
            old_F = F
            self._M_step(obs_flatten,lneta,lngamma,use_ext,multi=True)

        return self

    def _E_step(self,lnf,lnalpha,lnbeta,lneta,use_ext="F"):
        T = len(lnf)

        # Forward-Backward algorithm
        lnalpha, lnP_f = self._forward(lnf,lnalpha,use_ext)
        lnbeta, lnP_b = self._backward(lnf,lnbeta,use_ext)

        # check if forward and backward were done correctly
        dlnP = lnP_f-lnP_b
        if abs(dlnP) > 1.0e-6:
            print "warning forward and backward are not equivalent"

        # compute lneta for updating transition matrix
        if ext_imported and use_ext:
            if use_ext in ("c","C"):
                _hmmc._compute_lnEta_C(T,self._nstates,lnalpha,self._lnA, \
                    lnbeta,lnf,lnP_f,lneta)
            elif use_ext in ("f","F"):
                lneta = _hmmf.compute_lneta_f(lnalpha,self._lnA,lnbeta,lnf,lnP_f)
            else :
                raise ValueError, "ext_use must be either 'C' or 'F'"
        else:
            for i in xrange(self._nstates):
                for j in xrange(self._nstates):
                    for t in xrange(T-1):
                        lneta[t,i,j] = lnalpha[t,i] + self._lnA[i,j,] + \
                            lnf[t+1,j] + lnbeta[t+1,j]
            lneta -= lnP_f

        # compute lngamma for posterior on hidden states
        lngamma = lnalpha + lnbeta - lnP_f

        return lneta,lngamma,lnP_f

    def _M_step(self,obs,lneta,lngamma,use_ext="F",multi=False):
        self._calculate_sufficient_statistics(obs,lneta,lngamma,multi)
        self._update_parameters(obs,lneta,lngamma,multi)

    def _calculate_sufficient_statistics(self,obs,lneta,lngamma,multi=False):
        pass

    def _update_parameters(self,obs,lneta,lngamma,multi=False):
        if multi :
            # for multiple trajectories
            lg = np.vstack(lngamma)
            le = np.vstack(lneta)
            lngamma_sum = logsum(lg,0)
            #self._lnpi = lngamma_sum - logsum(lg)
            self._lnpi = logsum(np.array([lg_temp[0] for lg_temp in lngamma]),0)
            self._lnA = logsum(le,0) - logsum(np.vstack(\
              [lg_temp[:-1] for lg_temp in lngamma]),0)[:,np.newaxis]

        else:
            lngamma_sum = logsum(lngamma,0)
            #self._lnpi = lngamma_sum - logsum(lngamma_sum)
            # update initial probability
            self._lnpi = lngamma[0]
            # update transition matrix
            self._lnA = logsum(lneta,0) \
              - logsum(lngamma[:-1],0)[:,np.newaxis]

        return lngamma_sum

class MultinomialHMM(_BaseHMM):
    def __init__(self,N,M):
        _BaseHMM.__init__(self,N)
        self._mstates = M
        self._lnB = np.log(dirichlet([1.0]*M,N))

    def _log_like_f(self,obs):
        return self._lnB[:,obs].T

    def _complexity(self):
        """
        Count the number of parameter of HMM
        """
        comp = _BaseHMM._complexity(self) + float(self._nstates) * self._mstates
        return comp

    def simulate(self,T):
        pi_cdf = np.exp(self._lnpi).cumsum()
        A_cdf = np.exp(self._lnA).cumsum(1)
        B_cdf = np.exp(self._lnB).cumsum(1)
        z = np.zeros(T,dtype=np.int)
        o = np.zeros(T,dtype=np.int)
        r = random((T,2))
        z[0] = (pi_cdf > r[0,0]).argmax()
        o[0] = (B_cdf[z[0]] > r[0,1]).argmax()
        for t in xrange(1,T):
            z[t] = (A_cdf[z[t-1]] > r[t,0]).argmax()
            o[t] = (B_cdf[z[t]] > r[t,1]).argmax()

        return z,o

    def _update_parameters(self,obs,lneta,lngamma,multi=False):
        lngamma_sum = _BaseHMM._update_parameters(self,obs,lneta,lngamma)
        for j in xrange(self._mstates):
            self._lnB[:,j] = logsum(lngamma[obs==j,:],0) - lngamma_sum

class GaussianHMM(_BaseHMM):
    def __init__(self,N):
        _BaseHMM.__init__(self,N)

    def _initialize_HMM(self,obs,params="mc"):
        _BaseHMM._initialize_HMM(self,obs)
        nmix = self._nstates
        T,D = obs.shape
        if "m" in params:
            self.mu, temp = vq.kmeans2(obs,nmix)
        if "c" in params:
            self.cv = np.tile(np.identity(D),(self._nstates,1,1))

    def _log_like_f(self,obs):
        return log_like_Gauss(obs,self.mu,self.cv)

    def _complexity(self):
        """
        Count the number of parameter of HMM
        """
        nmix, ndim = self.mu.shape
        comp = _BaseHMM._complexity(self) + nmix * num_param_Gauss(ndim)
        return comp

    def simulate(self,T):
        N,D = self.mu.shape
        pi_cdf = np.exp(self._lnpi).cumsum()
        A_cdf = np.exp(self._lnA).cumsum(1)
        z = np.zeros(T,dtype=np.int)
        o = np.zeros((T,D))
        r = random(T)
        z[0] = (pi_cdf > r[0]).argmax()
        o[0] = sample_gaussian(self.mu[z[0]],self.cv[z[0]])
        for t in xrange(1,T):
            z[t] = (A_cdf[z[t-1]] > r[t]).argmax()
            o[t] = sample_gaussian(self.mu[z[t]],self.cv[z[t]])
        return z,o

    def _update_parameters(self,obs,lneta,lngamma,multi=False):
        lngamma_sum = _BaseHMM._update_parameters(self,obs,lneta,lngamma,multi)
        if multi:
            posteriors = np.exp(np.vstack(lngamma))
        else:
            posteriors = np.exp(lngamma)
        for k in xrange(self._nstates):
            post = posteriors[:, k]
            norm = 1.0 / post.sum()
            self.mu[k] = np.dot(post,obs) * norm
            avg_cv = np.dot(post * obs.T, obs) * norm
            self.cv[k] = avg_cv - np.outer(self.mu[k], self.mu[k])

test_model = GaussianHMM(3)
test_model.mu = np.array([[0.0,0.0],[1.0,3.0],[-3.0,0.0]])
test_model.cv = np.tile(np.identity(2),(3,1,1))
test_model.lnA = np.log([[0.9,0.05,0.05],[0.1,0.7,0.2],[0.1,0.4,0.5]])

if __name__ == "__main__":
      from sys import argv
      from scipy.linalg import eig
      ifreq = 10
      model = GaussianHMM(int(argv[1]))
      os = []
      zs = []
      for i in range(int(argv[2])):
            z,o = test_model.simulate(50)
            os.append(o)
            zs.append(z)
      o2 = np.vstack(os)
      if "-mult" in argv :
            model.fit_multi(os,ifreq=ifreq)
      else:
            model.fit(o2,ifreq=ifreq)
      print model.mu
      print model.cv
      print np.exp(model._lnpi)
      A = np.exp(model._lnA)
      e_val,e_vec = eig(A.T)
      print e_val.real
      print e_vec
