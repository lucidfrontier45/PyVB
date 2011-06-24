#!/usr/bin/python

import numpy as np
from scipy.cluster import vq
from util import logsum, log_like_Gauss2
from sampling import testData
from emgmm import EMGMM
from moments import *

class VBGMM(EMGMM):
    def __init__(self,nmix=10,u0=0.5,m0=0.0,beta0=1,nu0=1,V0=10.0):
        self._nstates = nmix
        self._u0 = np.ones(nmix) * u0
        self._nu0 = nu0
        self._V0 = V0
        self._beta0 = beta0
        self._m0 = m0

    def _init_prior(self,obs,adjust_prior=True):
        nobs, ndim = obs.shape
        if self._nu0 < ndim + 1:
            self._nu0 += ndim + 1

        if adjust_prior:
            self._m0 = obs.mean(0)
            self._V0 = np.cov(obs.T) * self._V0
        else:
            self._m0 = np.zeros(ndim)
            self._V0 = np.identity(ndim) * self._V0

    def _init_posterior(self,obs):
        nmix = self._nstates
        nobs, ndim = obs.shape
        avr_N = float(nobs) / float(nmix)
        self._u = np.ones(nmix) * (self._u0 + avr_N)
        self._nu = np.ones(nmix) * (self._nu0 + avr_N)
        self._beta = np.ones(nmix) * (self._beta0 + avr_N)
        self._V = np.tile(np.array(self._V0),(nmix,1,1))
        self._m, temp = vq.kmeans2(obs,nmix)

    def getExpectations(self):
        self.pi = E_pi_Dirichlet(self._u)
        self.mu = np.array(self._m)
        self.cv = self._V / self._nu[:,np.newaxis,np.newaxis]
        return self.pi, self.mu, self.cv

    def showModel(self,show_mu=False,show_cv=False,min_pi=0.01):
        """
        Obtain model parameters for relavent clusters
        """
        _ = self.getExpectations()
        return EMGMM.showModel(self,show_mu,show_cv,min_pi)

    def _log_like_f(self,obs):
        lnf = E_lnpi_Dirichlet(self._u)[np.newaxis,:] \
            + log_like_Gauss2(obs,self._nu,self._V,self._beta,self._m)
        return lnf

    def _KL_div(self):
        nmix = self._nstates
        KL = KL_Dirichlet(self._u,self._u0)
        for k in xrange(nmix):
            KL += KL_GaussWishart(self._nu[k],self._V[k],self._beta[k],\
                self._m[k],self._nu0,self._V0,self._beta0,self._m0)

        return KL

    def _E_step(self,obs):
        lnP = EMGMM._E_step(self,obs)
        KL = self._KL_div()
        L = lnP - KL
        return L

    def _update_parameters(self,min_cv=None):
        nmix = self._nstates
        self._u = self._u0 + self._N
        self._nu = self._nu0 + self._N
        self._beta = self._beta0 + self._N
        for k in xrange(nmix):
            self._m[k] = (self._beta0 * self._m0  \
                + self._N[k] * self._xbar[k]) / self._beta[k]
            dx = self._xbar[k] - self._m0
            self._V[k] = self._V0 + self._C[k] \
                + (self._N[k] * self._beta0 / self._beta[k]) * np.outer(dx,dx)

def test1(nmix=5):
    X = testData(5000)
    model = VBGMM(nmix)
    model.fit(X)
    #model.showModel()
    model.plot2d(X)

if __name__ == "__main__":
    from sys import argv
    nmix = int(argv[1])
    test1(nmix)
