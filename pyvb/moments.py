import numpy as np
from scipy.special import gammaln,digamma
from scipy.linalg import det, solve

# threshold for KL
_small_negative_number = -1.0e-10

def lnZ_Dirichlet(alpha):
    """
    log normalization constant of Dirichlet distribution
    input
      alpha [ndarray, shape (nmix)] : parameter of Dirichlet dist
    """

    Z = gammaln(alpha).sum() - gammaln(alpha.sum())
    return Z

def E_pi_Dirichlet(alpha):
    return alpha / alpha.sum()

def E_lnpi_Dirichlet(alpha):
    return digamma(alpha) - digamma(alpha.sum())

def KL_Dirichlet(alpha1,alpha2):
    """
    KL-div of Dirichlet distribution KL[q(alpha1)||p(alpha2)]
    input
      alpha1 [ndarray, shape (nmix)] : parameter of 1st Dirichlet dist
      alpha2 [ndarray, shape (nmix)] : parameter of 2nd Dirichlet dist
    """

    if len(alpha1) != len(alpha2) :
        raise ValueError, "dimension of alpha1 and alpha2 dont match"

    KL = - lnZ_Dirichlet(alpha1) + lnZ_Dirichlet(alpha2) \
        + np.dot((alpha1 - alpha2),(digamma(alpha1) - digamma(alpha1.sum())))

    if KL < _small_negative_number :
        raise ValueError, "KL must be larger than 0"

    return KL

def lnZ_Wishart(nu,V):
    """
    log normalization constant of Wishart distribution
    input
      nu [float] : dof parameter of Wichart distribution
      V [ndarray, shape (D x D)] : base matrix of Wishart distribution
      note <CovMat> = V/nu
    """
    if nu < len(V) + 1:
        raise ValueError, "dof parameter nu must larger than len(V)"

    D = len(V)
    lnZ = 0.5 * nu * (D * np.log(2.0) - np.log(det(V))) \
        + gammaln(np.arange(nu+1-D,nu+1)*0.5).sum()

    return lnZ

def E_lndetW_Wishart(nu,V):
    """
    mean of log determinant of precision matrix over Wishart <lndet(W)>
    input
      nu [float] : dof parameter of Wichart distribution
      V [ndarray, shape (D x D)] : base matrix of Wishart distribution
    """
    if nu < len(V) + 1:
        raise ValueError, "dof parameter nu must larger than len(V)"

    D = len(V)
    E = D*np.log(2.0) - np.log(det(V)) + \
        digamma(np.arange(nu+1-D,nu+1)*0.5).sum()

    return E

def KL_Wishart(nu1,V1,nu2,V2):
    """
    KL-div of Wishart distribution KL[q(nu1,V1)||p(nu2,V2)]
    """
    if nu1 < len(V1) + 1:
        raise ValueError, "dof parameter nu1 must larger than len(V1)"

    if nu2 < len(V2) + 1:
        raise ValueError, "dof parameter nu2 must larger than len(V2)"

    if len(V1) != len(V2):
        raise ValueError, \
            "dimension of two matrix dont match, %d and %d"%(len(V1),len(V2))

    D = len(V1)
    KL = 0.5 * ( (nu1 -nu2) * E_lndetW_Wishart(nu1,V1) \
        + nu1 * (np.trace(solve(V1,V2)) - D)) \
        - lnZ_Wishart(nu1,V1) + lnZ_Wishart(nu2,V2)

    if KL < _small_negative_number :
        print nu1,nu2,V1,V2
        raise ValueError, "KL must be larger than 0"

    return KL

def KL_GaussWishart(nu1,V1,beta1,m1,nu2,V2,beta2,m2):
    """
    KL-div of Gauss-Wishart distr KL[q(nu1,V1,beta1,m1)||p(nu2,V2,beta2,m2)
    """
    if len(m1) != len(m2):
        raise ValueError,  \
            "dimension of two mean dont match, %d and %d"%(len(m1),len(m2))

    D = len(m1)

    # first assign KL of Wishart
    KL1 = KL_Wishart(nu1,V1,nu2,V2)

    # the rest terms
    KL2 = 0.5 * (D * (np.log(beta1/float(beta2)) + beta2/float(beta1) - 1.0) \
        + beta2 * nu1 * np.dot((m1-m2),solve(V1,(m1-m2))))

    KL = KL1 + KL2

    if KL < _small_negative_number :
        raise ValueError, "KL must be larger than 0"

    return KL

def E_pi_StickBrake(tau):
    lntau = np.log(tau)
    lntau_sum = np.log(tau.sum(1)).cumsum()
    lnE = np.array(lntau[:,0])
    lnE[1:] += lntau[:-1,1].cumsum()
    lnE -= lntau_sum
    E = np.exp(lnE)
    return E

def E_lnpi_StickBrake(tau):
    lnzeta = digamma(tau) - digamma(tau.sum(1))[:,np.newaxis]
    lnzeta[1:,1] = lnzeta[:-1,1].cumsum()
    E = lnzeta.sum(1)
    return E

def KL_StickBrake(tau1,tau2):
    if len(tau1) != len(tau2):
        raise ValueError, "number of compenents didn't match"

    nmix = len(tau1)

    KL = np.sum([KL_Dirichlet(tau1[k],tau2[k]) for k in xrange(nmix)])

    if KL < _small_negative_number :
        raise ValueError, "KL must be larger than 0"
    
    return KL
