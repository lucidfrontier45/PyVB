import numpy as np
from numpy.random import randn
from scipy.linalg import inv, cholesky

def mixture_data(pi,m,cv,n=1000):
    """
    sample data from Gaussian Mixture
    input
        pi [ndarray, shape (nmix)] : mixing coefficients
        m [ndarray, shape (nmix x dim)] : parameter of mean
        S [ndarray, shape (nmix x dim x dim)] : parameter of cov matrix
        n [int] : sample number
    output
        X [ndarray, shape (n x nmix)] : sampled data
    """
    nmix,dim = m.shape
    pi2 = pi / np.sum(pi)
    x = []
    for k in xrange(nmix):
        nk = int(pi[k] * n)
        x.append(sample_gaussian(m[k],cv[k],nk))
    return np.concatenate(x)

def testData(n=1000):
    pi = np.array([0.2,0.5,0.3])
    m = np.array([[4.0,0],[-5,10],[0.0,2.0]])
    S = np.tile(np.identity(2),(3,1,1))
    return mixture_data(pi,m,S,n)

def sample_gaussian(m,cv,n=1):
    """Generate random samples from a Gaussian distribution.

    Parameters
    ----------
    m : array_like, shape (ndim)
        Mean of the distribution.

    cv : array_like, shape (ndim,ndim)
        Covariance of the distribution.

    n : int, optional
        Number of samples to generate. Defaults to 1.

    Returns
    -------
    obs : array, shape (ndim, n)
        Randomly generated sample
    """

    ndim = len(m)
    r = np.random.randn(n, ndim)
    if n == 1:
        r.shape = (ndim,)

    cv_chol = cholesky(cv)
    r = np.dot(r,cv_chol.T) + m

    return r

def sample_niw(mu_0,lmbda_0,kappa_0,nu_0):
    """
    Returns a sample from the normal/inverse-wishart distribution, conjugate prior for (simultaneously) unknown mean and unknown covariance in a Gaussian likelihood model. Returns covariance.
    """
    # p. 87 in Gelman

    # first sample Lambda ~ IW(lmbda_0^-1,nu_0)
    lmbda = inv(sample_wishart(inv(lmbda_0),nu_0))

    # then sample mu | Lambda ~ N(mu_0, Lambda/kappa_0)
    mu = np.random.multivariate_normal(mu_0,lmbda / kappa_0)

    return mu, lmbda


def sample_wishart(S, dof):
    """
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    Higher dof means higher 'variance' (larger eigenvalues) in precision space, meaning lower 'variance' in covariance space
    """
    n = S.shape[0]
    chol = cholesky(S)

    # use matlab's heuristic for choosing between the
    #   two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n))))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)