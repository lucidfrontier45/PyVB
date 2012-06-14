import numpy as np
from scipy.cluster import vq
from .util import normalize, logsum, log_like_Gauss, num_param_Gauss
from .sampling import testData

class EMGMM:
    """
    Gaussian Mixture Model with Expectation-Maximization (EM) Algorithm.

    Attributes
      _nstates [int] : number of hidden states, nmix
      pi [ndarray, shape (_nstates)] : mixing coefficients
      mu [ndarray, shape (_nstates, dim)] : mean vectors
      cv [ndarray, shape (_nstates, dim, dim)] : covariance matrix
    Methods
      showModel : show model parameters
      eval_hidden_states : get the propability of hidden states
      score : score the model with respect to some criteria
      fit : fit model parameters
      decode : return most probable hidden states
      plot1d : plot most probable hidden states along one axis of data
      plot2d : plot most probable hidden states along two axes of data
      makeTransMat : make transition matrix by regarding the data as time series
    """
    def __init__(self,nmix=10):
        # maximum number of the hidden clusters
        self._nstates = nmix

    def _init_params(self,obs,adjust_prior=True):
        """
        Initialize prior and posterior parameters before running
        iterative fitting.
        """
        nmix = self._nstates
        nobs, ndim = obs.shape
        self._init_prior(obs,adjust_prior)
        self._init_posterior(obs)

        # auxuality variable to store unnormalized sample posterior cov mat
        self._C = np.empty((nmix,ndim,ndim))

    def _init_prior(self,obs,adjust_prior):
        pass

    def _init_posterior(self,obs):
        """
        Initialize posterior parameters
        """
        nmix = self._nstates
        nobs, ndim = obs.shape
        # initialize hidden states
        self.z = np.ones((nobs,nmix)) / float(nmix)
        # initialize mixing coefficients
        self.pi = np.ones(nmix) / float(nmix)
        # initialize mean vectors with K-Means clustering
        self.mu, temp = vq.kmeans2(obs,nmix)
        # initialize covariance matrices with sample covariance matrix
        self.cv = np.tile(np.atleast_2d(np.cov(obs.T)),(nmix,1,1))

    def showModel(self,show_mu=False,show_cv=False,min_pi=0.01):
        """
        Obtain model parameters for relavent clusters
        input
          show_mu [bool] : if print mean vectors
          show_cv [bool] : if print covariance matrices
          min_pi [float] : components whose pi < min_pi will be excluded
        output
          relavent_clusters [list of list] : a list of list whose fist index is
          the cluster and the second is properties of each cluster.
            - relavent_clusters[i][0] = mixing coefficients
            - relavent_clusters[i][1] = cluster id
            - relavent_clusters[i][2] = mean vector
            - relavent_clusters[i][3] = covariance matrix
          Clusters are sorted in descending order along their mixing coeffcients
        """
        nmix = self._nstates

        # make a tuple of properties and sort its member by mixing coefficients
        params = sorted(zip(self.pi,range(nmix),self.mu,self.cv),\
            key=lambda x:x[0],reverse=True)

        relavent_clusters = []
        for k in xrange(nmix):
            # exclude clusters whose pi < min_pi
            if params[k][0] < min_pi:
                break

            relavent_clusters.append(params[k])
            print "\n%dth component, pi = %8.3g" % (k,params[k][0])
            print "cluster id =", params[k][1]
            if show_mu:
                print "mu =",params[k][2]
            if show_cv:
                print "cv =",params[k][3]

        return relavent_clusters

    def _log_like_f(self,obs):
        """
        log-likelihood function of of complete data, lnP(X,Z|theta)
        input
          obs [ndarray, shape (nobs,ndim)] : observed data
        output
          lnf [ndarray, shape (nobs, nmix)] : log-likelihood
            where lnf[n,k] = lnP(X_n,Z_n=k|theta)
        """
        lnf = np.log(self.pi)[np.newaxis,:] \
            + log_like_Gauss(obs,self.mu,self.cv)
        return lnf

    def eval_hidden_states(self,obs):
        """
        Calc P(Z|X,theta) = exp(lnP(X,Z|theta)) / C
            where C = sum_Z exp(lnP(X,Z|theta)) = P(X|theta)
        input
          obs [ndarray, shape (nobs,ndim)] : observed data
        output
          z [ndarray, shape (nobs,nmix)] : posterior probabiliry of
            hidden states where z[n,k] = P(Z_n=k|X_n,theta)
          lnP [float] : lnP(X|theta)
        """
        lnf = self._log_like_f(obs)
        lnP = logsum(lnf,1)
        z = np.exp(lnf - lnP[:,np.newaxis])
        return z,lnP.sum()

    def score(self,obs,mode="BIC"):
        """
        score the model
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          mode [string] : one of 'ML', 'AIC' or 'BIC'
        output
          S [float] : score of the model
        """
        z,lnP = self.eval_hidden_states(obs)
        nmix = self._nstates
        nobs, ndim = obs.shape
        k = num_param_Gauss(ndim)
        if mode in ("AIC", "aic"):
            # use Akaike information criterion
            S = -lnP + (nmix + k * nmix)
        if mode in ("BIC", "bic"):
            # use Bayesian information criterion
            S = -lnP + (nmix + k * nmix) * np.log(nobs)
        else:
            # use negative likelihood
            S = -lnP
        return S

    def fit(self,obs,niter=1000,eps=1.0e-4,ifreq=10,init=True,plot=False):
        """
        Fit model parameters via EM algorithm
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          niter [int] : maximum number of iteration cyles
          eps [float] : convergence threshold
          ifreq [int] : frequency of printing fitting process
          init [bool] : flag for initialization
          plot [bool] : flag for plotting the result
        """

        # initialize parameters
        if init:
            self._init_params(obs)

        # initialize free energy
        F = 1.0e50

        # main loop
        for i in xrange(niter):

            # performe E step and set new free energy
            F_new = - self._E_step(obs)

            # take difference
            dF = F_new - F

            # check convergence
            if abs(dF) < eps :
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" \
                %(i,F_new,dF)
                print "%12.6e < %12.6e Converged" %(dF, eps)
                break

            # print iteration info
            if i % ifreq == 0 and dF < 0.0:
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e" \
                %(i,F_new,dF)
            elif dF > 0.0:
                print "%8dth iter, Free Energy = %12.6e, dF = %12.6e warning" \
              %(i,F_new,dF)

            # update free energy
            F = F_new

            # update parameters by M step
            self._M_step(obs)

        # plot clustering result
        if plot:
            self.plot2d(obs)

        return self

    def _E_step(self,obs):
        """
        E step, calculate posteriors P(Z|X,theta)
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
        output
          lnP [float] : log-likelihood, lnP(X|theta)
        """
        self.z, lnP = self.eval_hidden_states(obs)
        return lnP

    def _M_step(self,obs):
        """
        M step calculates sufficient statistics and use them
         to update parameters
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
        """
        self._calculate_sufficient_statistics(obs)
        self._update_parameters()

    def _calculate_sufficient_statistics(self,obs):
        """
        Calculate sufficient Statistics
        """
        nmix = self._nstates

        # posterior average number of observation
        self._N = self.z.sum(0)
        # posterior average of x
        self._xbar = np.dot(self.z.T,obs) / self._N[:,np.newaxis]
        # posterior unnormalized sample covariance matrices
        for k in xrange(nmix):
            dobs = obs - self._xbar[k]
            self._C[k] = np.dot((self.z[:,k] * dobs.T), dobs)

    def _update_parameters(self,min_cv=0.001):
        """
        Update parameters of posterior distribution by precomputed
            sufficient statistics
        """
        nmix = self._nstates
        # parameter for mixing coefficients
        self.pi = self._N / self._N.sum()
        # parameters for mean vectors
        self.mu = np.array(self._xbar)
        # parameters for covariance matrices
        self.cv = np.identity(len(self._C[0])) * min_cv \
            + self._C / self._N[:,np.newaxis,np.newaxis]

    def decode(self,obs,eps=0.01):
        """
        Return most probable cluster ids.
        Clusters are sorted along the mixing coefficients
        """
        # get probabilities of hidden states
        z,lnP = self.eval_hidden_states(obs)
        # take argmax
        codes = z.argmax(1)
        # get sorted ids
        params = self.showModel(min_pi=eps)
        # assign each observation to corresponding cluster
        clust_pos = []
        for p in params:
            clust_pos.append(codes==p[1])
        return clust_pos

    def plot1d(self,obs,d1=0,eps=0.01,clust_pos=None):
        """
        plot data of each cluster along one axis
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          d1 [int, optional] : id of axis
          clust_pos [list, optional] : decoded cluster postion
        """
        # plotting symbols
        symbs = ".hd^x+"
        # plot range
        l = np.arange(len(obs))
        # decode observed data
        if clust_pos == None:
            clust_pos = self.decode(obs,eps)
        # import pyplot
        try :
            import matplotlib.pyplot as plt
        except ImportError :
            print "cannot import pyplot"
            return
        # plot data
        for k,pos in enumerate(clust_pos):
            symb = symbs[k / 7]
            plt.plot(l[pos],obs[pos,d1],symb,label="%3dth cluster"%k)
        plt.legend(loc=0)
        plt.show()

    def plot2d(self,obs,d1=0,d2=1,eps=0.01,clust_pos=None):
        """
        plot data of each cluster along two axes
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          d1 [int, optional] : id of the 1st axis
          d2 [int, optional] : id of the 2nd axis
          clust_pos [list, optional] : decoded cluster postion
        """
        symbs = ".hd^x+"
        if clust_pos == None:
            clust_pos = self.decode(obs,eps)
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

    def makeTransMat(self,obs,norm=True,min_degree=1,eps=0.01):
        """
        Make transition probability matrix MT
          where MT[i,j] = N(x_{t+1}=j|x_t=i)
        input
          obs [ndarray, shape(nobs,ndim)] : observed data
          norm [bool] : if normalize or not
          min_degree [int] : transitions which occured less than min_degree
            times will be omitted.
        output
          MT [ndarray, shape(nmix,nmix)] : transition probabiliry matrix
            nmix is effective number of clusters
        """
        # get probability of hidden states
        z,lnP = self.eval_hidden_states(obs)
        dim = self._nstates

        #initialize MT
        MT = np.zeros((dim,dim))

        # main loop
        for t in xrange(1,len(z)-1):
            MT += np.outer(z[t-1],z[t])

        for i in xrange(len(MT)):
            for j in xrange(len(MT)):
                if MT[i,j] < min_degree:
                    MT[i,j] = 0.0

        # extract relavent cluster
        params = self.showModel(min_pi=eps)
        cl = [p[1] for p in params]
        MT = np.array([mt[cl] for mt in MT[cl]])

        if norm:
            # MT[i,j] = P(x_{t+1}=j|x_t=i)
            MT = normalize(MT,1)

        return MT
