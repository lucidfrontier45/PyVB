import numpy as np
from scipy.cluster import vq
from .util import log_like_Gauss2
from .sampling import testData
from .vbgmm import VBGMM
from .moments import *

class DPGMM(VBGMM):
    def __init__(self,nmix=10,alpha=0.5,m0=0.0,beta0=1,nu0=1,V0=10.0):
        # maximum number of the hidden clusters
        self._nstates = nmix
        # hyper parameter for Stick-Braking prior mixing coefficients
        self._tau0 = np.ones((nmix,2))
        self._tau0[:,1] *= alpha
        # hyper parameters for prior precision matrix
        self._nu0 = nu0
        self._V0 = V0
        # hyperparameters for prior mean vector
        self._beta0 = beta0
        self._m0 = m0
        
    def _init_posterior(self,obs):
        """
        Initialize posterior parameters
        """
        nmix = self._nstates
        nobs, ndim = obs.shape
        avr_N = float(nobs) / float(nmix)
        # parameters of posterior mixing coefficients
        self._tau = np.array(self._tau0)
        self._tau[:,0] += avr_N
        self._tau[:,1] += (nobs - (np.ones(nmix)*avr_N).cumsum())
        # parameters of posterior precision matrices
        self._nu = np.ones(nmix) * (self._nu0 + avr_N)
        self._V = np.tile(np.array(self._V0),(nmix,1,1))
        # parameters of posterior mean vectors
        self._beta = np.ones(nmix) * (self._beta0 + avr_N)
        self._m, temp = vq.kmeans2(obs,nmix) # initialize by K-Means
        
    def getExpectations(self):
        """
        Calculate expectations of parameters over posterior distribution
        """
        # <pi_k>_Q(pi_k)
        self.pi = E_pi_StickBrake(self._tau)

        # <mu_k>_Q(mu_k,W_k)
        self.mu = np.array(self._m)

        # inv(<W_k>_Q(W_k))
        self.cv = self._V / self._nu[:,np.newaxis,np.newaxis]

        return self.pi, self.mu, self.cv
        
    def _log_like_f(self,obs):
        """
        mean log-likelihood function of of complete data over posterior
            of parameters, <lnP(X,Z|theta)>_Q(theta)
        input
          obs [ndarray, shape (nobs,ndim)] : observed data
        output
          lnf [ndarray, shape (nobs, nmix)] : log-likelihood
            where lnf[n,k] = <lnP(X_n,Z_n=k|theta_k)>_Q(theta_k)
        """

        lnf = E_lnpi_StickBrake(self._tau)[np.newaxis,:] \
            + log_like_Gauss2(obs,self._nu,self._V,self._beta,self._m)
        return lnf
    
    def _KL_div(self):
        """
        Calculate KL-divergence of parameter distribution KL[Q(theta)||P(theta)]
        output
          KL [float] : KL-div
        """
        nmix = self._nstates

        # first calculate KL-div of mixing coefficients
        KL = KL_StickBrake(self._tau,self._tau0)

        # then calculate KL-div of mean vectors and precision matrices
        for k in xrange(nmix):
            KL += KL_GaussWishart(self._nu[k],self._V[k],self._beta[k],\
                self._m[k],self._nu0,self._V0,self._beta0,self._m0)
        return KL
         
    def _update_parameters(self,min_cv=None):
        """
        Update parameters of variational posterior distribution by precomputed
            sufficient statistics
        """

        nmix = self._nstates
        # parameter for mixing coefficients
        self._tau[:,0] = self._tau0[:,0] + self._N
        self._tau[:,1] = self._tau0[:,1] + (self._N.sum() - self._N.cumsum())

        # parameters for mean vectors and precision matrices
        # scalar parameters of Gauss-Wishart
        self._nu = self._nu0 + self._N
        self._beta = self._beta0 + self._N
        # vector or matrix parameters of Gauss-Wishart
        for k in xrange(nmix):
            self._m[k] = (self._beta0 * self._m0  \
                + self._N[k] * self._xbar[k]) / self._beta[k]
            dx = self._xbar[k] - self._m0
            self._V[k] = self._V0 + self._C[k] \
                + (self._N[k] * self._beta0 / self._beta[k]) * np.outer(dx,dx)
