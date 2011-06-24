#!/usr/bin/python

import numpy as np
from scipy.cluster import vq
from util import normalize, logsum, log_like_Gauss
from sampling import testData

class EMGMM:
    """
    Gaussian Mixture Model with Expectation-Maximization (EM) Algorithm.
    """ 
    def __init__(self,nmix=10):
        # maximum number of the hidden clusters
        self._nstates = nmix
    
    def _init_params(self,obs,adjust_prior=True):
        nmix = self._nstates
        nobs, ndim = obs.shape
        self._init_prior(obs,adjust_prior)
        self._init_posterior(obs)
        
        self._C = np.empty((nmix,ndim,ndim))

    def _init_prior(self,obs,adjust_prior):
        pass
    
    def _init_posterior(self,obs):
        nmix = self._nstates
        nobs, ndim = obs.shape
        self.z = np.ones((nobs,nmix)) / float(nmix)
        self.pi = np.ones(nmix) / float(nmix)
        self.mu, temp = vq.kmeans2(obs,nmix)
        self.cv = np.tile(np.cov(obs.T),(nmix,1,1)) * 10.0
          
    def showModel(self,show_mu=False,show_cv=False,min_pi=0.01):
        """
        Obtain model parameters for relavent clusters
        """
        nmix = self._nstates
        params = sorted(zip(self.pi,range(nmix),self.mu,self.cv),\
            key=lambda x:x[0],reverse=True)
        relavent_clusters = []
        for k in xrange(nmix):
            if params[k][0] < min_pi:
                break
            relavent_clusters.append(params[k])
            print "\n%dth component, pi = %8.3g" % (k,params[k][0])
            print "cluster id =", params[k][1]
            if show_mu:
                print "mu =",params[k][2]
            if show_cv:
                print "tau =",params[k][3]
        return relavent_clusters

    def _log_like_f(self,obs):
        lnf = np.log(self.pi)[np.newaxis,:] \
            + log_like_Gauss(obs,self.mu,self.cv) 
        return lnf

    def eval_hidden_states(self,obs):
        lnf = self._log_like_f(obs)
        lnP = logsum(lnf,1)
        z = np.exp(lnf - lnP[:,np.newaxis])
        return z,lnP.sum()
    
    def fit(self,obs,niter=1000,eps=1.0e-4,ifreq=10,init=True,plot=False):
        if init:
            self._init_params(obs)
        
        F = 1.0e50
        
        for i in xrange(niter):
            F_new = - self._E_step(obs)
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
        if plot:
            self.plot2d(obs)
        return self
    
    def _E_step(self,obs):
        self.z, lnP = self.eval_hidden_states(obs)      
        return lnP
        
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
        
    def _update_parameters(self,min_cv=0.001):
        nmix = self._nstates
        self.pi = self._N / self._N.sum()
        self.mu = np.array(self._xbar)
        self.cv = np.identity(len(self._C[0])) * min_cv \
            + self._C / self._N[:,np.newaxis,np.newaxis]

    def decode(self,obs):
        z,lnP = self.eval_hidden_states(obs)
        codes = z.argmax(1)
        params = self.showModel()
        clust_pos = []
        for p in params:
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
        
    def makeTransMat(self,obs,norm=True,min_degree=10):
        # MT[i,j] = N(x_{t+1}=j|x_t=i)
        z,lnP = self.eval_hidden_states(obs)
        dim = self._nstates
        MT = np.zeros((dim,dim))
        for t in xrange(1,len(z)-1):
            MT += np.outer(z[t-1],z[t])
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


def test1(n=1000):
    X = testData(n)
    model = EMGMM(10)
    model.fit(X)
    model.showModel()
    
if __name__ == "__main__":
    from sys import argv
    test1(int(argv[1]))
