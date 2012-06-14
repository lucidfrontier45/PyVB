import numpy as np
from scipy.cluster import vq
from scipy.special import digamma
from .vbhmm import _BaseVBHMM

from .util import log_like_Gauss2
from .moments import *

# import Fortran95 extension module
from . import _hmmf
extF_imported = True
default_ext = "F"


class _BaseMEHMM(_BaseVBHMM):
    """
    This is a base class for HMM with Maximization Expectation Learning
    All ME-HMM should be an inheritant of this class
    """
      
    def eval_hidden_states(self,obs,use_ext=default_ext,multi=False):
        """
        Performe one Estep.
        Then obtain variational free energy and posterior over hidden states
        """
#        if multi :
#            z = []
#            lnP = []
#            for o in obs:
#                zz, ln = self.eval_hidden_states(o,use_ext,multi=False)
#                z.append(zz)
#                lnP.append(ln)
#            z = np.vstack(z)
#            lnP = np.sum(lnP)
#        else :
        if True:
            lnf = self._log_like_f(obs)
            z, lnP = self._E_step(lnf,use_ext)
            return z, lnP    
            
    def decode(self,obs,use_ext=default_ext,multi=False):
        """
        Get the most probable cluster id
        """
        z,lnP = self.eval_hidden_states(obs,use_ext,multi)
        return z
        
    def fit(self,obs,niter=10000,eps=1.0e-6,ifreq=10,\
            init=True,use_ext=default_ext):
        """
        Fit the HMM via VB-EM algorithm
        """
        if init:
            self._initialize_HMM(obs)
            old_F = 1.0e20
            print set(self.z)            
            
        for i in xrange(niter):
            # VB-E step
            lnf = self._log_like_f(obs)
            self.z, lnP = self._E_step(lnf,use_ext)

            # check convergence
            KL = self._KL_div()
            F = -lnP + KL
            dF = F - old_F
    
            if(abs(dF) < eps):
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F,dF)
                print "%12.6e < %12.6e Converged" %(dF, eps)
                break
            if i % ifreq == 0 and dF < 0.0:
                print "%6dth iter, F = %15.8e  df = %15.8e"%(i,F,dF)
            elif dF >= 0.0:
                print "%6dth iter, F = %15.8e  df = %15.8e warning"%(i,F,dF)
        
            old_F = F

            # update parameters via VB-M step
            self._M_step(obs,multi=False)

        return self

#    def fit_multi(self,obss,niter=1000,eps=1.0e-4,ifreq=10,\
#            init=True,use_ext=default_ext):
#        """
#        Fit HMM via VB-EM algorithm with multiple trajectories
#        """
#        nobss = len(obss) # number of trajectories
#        nobs = [len(obs) for obs in obss] # numbers of observations in all trajs
#        i_max_obs = np.argmax(nobs)
#        obs_flatten = np.vstack(obss) # flattened observations (sum(nobs)xdim)
#        nmix = self._nstates
#    
#        # get posistion id for each traj
#        # i.e. obss[i] = obs[pos_ids[i][0]:pos_ids[i][1]]
#        pos_ids = [] 
#        j = 0
#        for i in xrange(nobss):
#            pos_ids.append((j,j+nobs[i]))
#            j += nobs[i]
#    
#        if init:
#            self._initialize_HMM(obs_flatten)
#            old_F = 1.0e20   
#    
#        # allocate space for forward-backward
#        lneta = []
#        lngamma = []
#        for nn in xrange(nobss):
#            lneta.append(np.zeros((len(obss[nn])-1,nmix,nmix)))
#            lngamma.append(np.zeros((len(obss[nn]),nmix)))
#        lnalpha, lnbeta, lneta_temp = self._allocate_temp(obss[i_max_obs])
#    
#        for i in xrange(niter):
#            lnP = 0.0
#            lnf = self._log_like_f(obs_flatten)
#            for nn in xrange(nobss):
#                Ti,Tf = pos_ids[nn]
#                e, g, p = self._E_step(lnf[Ti:Tf],lnalpha[:nobs[nn]],\
#                    lnbeta[:nobs[nn]],lneta_temp[:nobs[nn]-1],use_ext)
#                lneta[nn] = e[:]
#                lngamma[nn] = g[:]
#                lnP += p
#    
#            KL = self._KL_div()
#            #print -lnP,KL
#            F = -lnP + KL
#            dF = F - old_F
#            if(abs(dF) < eps):
#                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" %(i,F,dF)
#                print "%12.6e < %12.6e Converged" %(dF, eps)
#                break
#
#            if i % ifreq == 0 and dF < 0.0:
#                print "%6dth iter, F = %15.8e  df = %15.8e"%(i,F,dF)
#            elif dF >= 0.0:
#                print "%6dth iter, F = %15.8e  df = %15.8e warning"%(i,F,dF)
#
#            old_F = F
#            self._M_step(obs_flatten,np.vstack(lneta),lngamma,multi=True)
#    
#        return self

    def _E_step(self,lnf,use_ext=default_ext):
        z, lnP = _hmmf.viterbi_f(self._lnpi,self._lnA,lnf)
        return np.array(z,dtype=np.int32), lnP  
        
    def _M_step(self,obs,multi=False):
        self._calculate_sufficient_statistics(obs,multi)
        self._update_parameters(obs,multi)
        
    def _calculate_sufficient_statistics(self,obs,multi=False):
        pass
#        if multi:
#            self.z = np.exp(np.vstack(lngamma))
#            self.z0 = np.exp([lg[0] for lg in lngamma]).sum(0)
#        else:
#            # z[n,k] = Q(Zn=k)
#            self.z = np.exp(lngamma)

    def _update_parameters(self,obs,multi=False):

        # update parameters of initial prob 
        
#        if multi :
#            self._WPi = self._uPi + self.z0
#        else:
#            #self.WPi = self._uPi + self.z.sum(0)
#            self._WPi = self._uPi + self.z[0]

        self._WPi = self._uPi.copy()
        self._WPi[self.z[0]] += 1      
        self._lnpi = E_lnpi_Dirichlet(self._WPi)
        
        # update parameters of transition prob 
        self._WA = self._uA.copy()

        # extract N(z[t]=k,z[t+1]=k')
        for k in xrange(self._nstates):
            
            # the position of z[t+1] s.t. z[t] == k
            mask = np.where(self.z == k)[0] + 1
            
            if len(mask) == 0:
                continue
            
            # remove terminal if exceeded total samples
            if mask[-1] == len(self.z):
                mask = mask[:-1]
            
            # update WA
            n_trans = np.bincount(self.z[mask])
            self._WA[k,:len(n_trans)] += n_trans 
                
        for k in xrange(self._nstates):
            self._lnA[k,:] = E_lnpi_Dirichlet(self._WA[k,:])
        
#class VBMultinomialHMM(_BaseVBHMM):
#    def __init__(self,N,M,uPi0=0.5,uA0=0.5,uB0=0.5):
#        _BaseVBHMM.__init__(self,N,uPi0,uA0)
#        self._mstates = M
#        self._uB = np.ones((N,M)) * uB0
#        self._WB = np.array(self._uB)
#        self._lnB = np.log(dirichlet([1.0]*M,N))
#      
#    def _log_like_f(self,obs):
#        return self._lnB[:,obs].T
#
#    def simulate(self,T):
#        pass
#
#    def _update_parameters(self,obs,lneta,lngamma):
#        _BaseVBHMM._update_parameters(self,obs,lneta,lngamma)
#        for j in xrange(self._mstates):
#            self._WB[:,j] = self._uB[:,j] + self.z[obs==j,:].sum(0)
#            self._lnB[:,j] = digamma(self._WB[:,j]) \
#                    - digamma(self._WB[:,j].sum())

class MEGaussianHMM(_BaseMEHMM):
    """
    ME-HMM with Gaussian emission probability.
    ME-E step is Viterbi Algorithm.
    Parameter estimation is almost same as VBGMM.
    """
    def __init__(self,N,uPi0=0.5,uA0=0.5,m0=0.0,beta0=1,nu0=1,s0=0.01):
        _BaseMEHMM.__init__(self,N,uPi0,uA0)
        self._m0 = m0
        self._beta0 = beta0
        self._nu0 = nu0
        self._s0 = s0
        self._N = np.zeros(N)
        
    def _initialize_HMM(self,obs,params="ms",scale=10.0):
        _BaseMEHMM._initialize_HMM(self,obs)
        nmix = self._nstates
        T,D = obs.shape
        if self._nu0 < D:
            self._nu0 += D
        if "m" in params:
            self._m0 = np.mean(obs,0)
        if "s" in params:
            self._V0 = np.atleast_2d(np.cov(obs.T,bias=1)) * scale

        # for mean vector and hidden states
        self._m, self.z = vq.kmeans2(obs,nmix)
        count = np.bincount(self.z)
        self._N[:len(count)] = count[:]
        self._beta = self._beta0 + self._N
        # for covarience matrix
        self._V = np.tile(np.array(self._V0),(nmix,1,1))
        self._nu = self._nu0 + self._N

        # aux
        self._xbar = np.array(self._m)
        self._C = np.array(self._V)
    
    def _log_like_f(self,obs):
        return log_like_Gauss2(obs,self._nu,self._V,self._beta,self._m)
        
    def _KL_div(self):
        nmix = self._nstates
        KL = _BaseVBHMM._KL_div(self)
        for k in xrange(nmix):
            KLg = KL_GaussWishart(self._nu[k],self._V[k],self._beta[k],\
                self._m[k],self._nu0,self._V0,self._beta0,self._m0)
            KL += KLg
        return KL
    
    def _calculate_sufficient_statistics(self,obs,multi=False):
        nmix = self._nstates
        T,D = obs.shape

        count = np.bincount(self.z)
        self._N[:] = 0
        self._N[:len(count)] = count[:]
        
        for k in xrange(nmix):
            mask = (self.z == k)
            
            # ignore empty cluster 
            if self._N[k] == 0:
                self._xbar[k] = 0.0
                self._C[k] = 0.0

            else :
                self._xbar[k] = np.mean(obs[mask],0)
                self._C[k] = np.atleast_2d(
                        np.cov(obs[mask].T,bias=1)) * self._N[k] 

    def _update_parameters(self,obs,multi=False):
        nmix = self._nstates
        T,D = obs.shape
        _BaseMEHMM._update_parameters(self,obs,multi)
        self._beta = self._beta0 + self._N
        self._nu = self._nu0 + self._N
        self._V = self._V0 + self._C
        for k in xrange(nmix):
            if self._N[k] == 0:
                self._m[k] = self._m0[:]
                self._V[k] = self._V0[:]
            else:                
                self._m[k] = (self._beta0 * self._m0 + self._N[k] * self._xbar[k])\
                            / self._beta[k]
                dx = self._xbar[k] - self._m0
                self._V[k] += (self._beta0 * self._N[k] / self._beta[k]) \
                    * np.outer(dx, dx)
                    
    def getExpectations(self):
        """
        Calculate expectations of parameters over posterior distribution
        """
        _BaseMEHMM.getExpectations(self)
        # <mu_k>_Q(mu_k,W_k)
        self.mu = np.array(self._m)

        # inv(<W_k>_Q(W_k))
        self.cv = self._V / self._nu[:,np.newaxis,np.newaxis]     
        
        return self.pi, self.A, self.mu, self.cv
                
        
    def showModel(self,show_pi=True,show_A=True,show_mu=False,\
        show_cv=False,eps=1.0e-2):
        ids, pi, A = _BaseMEHMM.showModel(self,show_pi,show_A,eps)
        mu = self.mu[ids]
        cv = self.cv[ids]
        for k in range(len(ids)):
            i = ids[k]
            print "\n%dth component, pi = %8.3g" % (k,self.pi[i])
            print "cluster id =", i
            if show_mu:
                print "mu =",self.mu[i]
            if show_cv:
                print "cv =",self.cv[i]      
        return ids,pi,A,mu,cv

    def getClustPos(self,obs,use_ext=default_ext,multi=False,eps=1.0e-2):
        ids,pi,A,m,cv = self.showModel(eps=eps)
        codes = self.decode(obs,use_ext,multi)
        clust_pos = []
        for k in ids:
            clust_pos.append(codes==k)
        return clust_pos

    def plot1d(self,obs,d1=0,eps=0.01,use_ext=default_ext,\
            multi=False,clust_pos=None):
        symbs = ".hd^x+"
        if multi :
            obs2 = np.vstack(obs)
        else :
            obs2 = obs
            
        l = np.arange(len(obs2))
        if clust_pos == None:
            clust_pos = self.getClustPos(obs,use_ext,multi,eps)
        try :
            import matplotlib.pyplot as plt
        except ImportError :
            print "cannot import pyplot"
            return
        for k,pos in enumerate(clust_pos):
            symb = symbs[k / 6]
            plt.plot(l[pos],obs2[pos,d1],symb,label="%3dth cluster"%k)
        plt.legend(loc=0)
        plt.show()

    def plot2d(self,obs,d1=0,d2=1,eps=0.01,use_ext=default_ext,\
            multi=False,clust_pos=None):
        symbs = ".hd^x+"
        if multi :
            obs2 = np.vstack(obs)
        else :
            obs2 = obs
            
        if clust_pos == None:
            clust_pos = self.getClustPos(obs,use_ext,multi,eps)
        try :
            import matplotlib.pyplot as plt
        except ImportError :
            print "cannot import pyplot"
            return
        for k,pos in enumerate(clust_pos):
            symb = symbs[k / 6]
            plt.plot(obs2[pos,d1],obs2[pos,d2],symb,label="%3dth cluster"%k)
        plt.legend(loc=0)
        plt.show()
