import numpy as np
from scipy.special import gammaln,digamma
from scipy.linalg import det, solve

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

    if KL < 0.0 :
        raise ValueError, "KL must be larger than 0"

    return KL

def lnZ_Wishart(nu,V):
    if nu < len(V) + 1:
        raise ValueError, "dof parameter nu must larger than len(V)"
    
    D = len(V)
    lnZ = 0.5 * nu * (D * np.log(2.0) - np.log(det(V))) \
        + gammaln(np.arange(nu+1-D,nu+1)*0.5).sum()
    
    return lnZ
    
def E_lndetW_Wishart(nu,V):
    if nu < len(V) + 1:
        raise ValueError, "dof parameter nu must larger than len(V)"
    
    D = len(V)
    E = D*np.log(2.0) - np.log(det(V)) + \
        digamma(np.arange(nu+1-D,nu+1)*0.5).sum()
    
    return E
    
def KL_Wishart(nu1,V1,nu2,V2):
    if nu1 < len(V1) + 1:
        raise ValueError, "dof parameter nu1 must larger than len(V1)"
        
    if nu2 < len(V2) + 1:
        raise ValueError, "dof parameter nu2 must larger than len(V2)"
    
    if len(V1) != len(V2):
        raise ValueError, \
            "dimension of two matrix dont match, %d and %d"%(len(V1),len(V2))
    
    D = len(V1)
    KL = 0.5 * ( (nu1 -nu2) * E_lndetW_Wishart(nu1,V1) \
        - nu1 * (np.trace(solve(V1,V2)) - D)) \
        - lnZ_Wishart(nu1,V1) + lnZ_Wishart(nu2,V2)
        
    if KL < 0.0 :
        raise ValueError, "KL must be larger than 0"
        
    return KL
    
def KL_GaussWishart(nu1,V1,beta1,m1,nu2,V2,beta2,m2):
    if len(m1) != len(m2):
        raise ValueError,  \
            "dimension of two mean dont match, %d and %d"%(len(m1),len(m2))
    
    D = len(m1)

    # first assign KL of Wishart    
    KL = KL_Wishart(nu1,V1,nu2,V2)
    
    # the rest terms
    KL += 0.5 * (D * (np.log(beta1/beta2) + beta2/beta1 - 1.0) + \
        beta2 * nu1 * np.dot((m1-m2),solve(V1,(m1-m2))))
    
    if KL < 0.0 :
        raise ValueError, "KL must be larger than 0"
        
    return KL