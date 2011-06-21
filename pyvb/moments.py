import numpy as np
from scipy.special import gammaln,digamma
from scipy.linalg import inv, det, solve
from scikits.learn.mixture import lmvnpdf

def lmvnpdf2(obs,m,cv,nu,beta):
  """
  mean log-likelihood of multivariate Gaussian
  <lnP(X|Z,mu,Sigma)>_Q(Z,mu,Sigma)
  This function need "lmnvpdf(full)" from scikits.learn.mixture

  input
    obs [ndarray, shape (T x D)] : observed data
    m [ndarray, shape (nmix x D) : mean of mean vector over Q(mu,Sigma)
    cv [ndarray, shape (nmix x D x D)] : mean of covariance matrix over Q
    nu [ndarray, shape (nmix)] : dof parameter for Wishart dist
    beta [ndarray, shape(nmix)] : scale parameter for Gauss-Wishart
  output
    lnf [ndarray, shape (T x nmix)] : mean log-likelihood
  """

  nmix = len(m)
  T,D = obs.shape
  lnf = lmvnpdf(obs,m,cv,"full")
  for k in xrange(nmix):
    temp = -0.5* (D *(np.log(0.5*nu[k]) + 1.0/beta[k]) - \
        digamma(np.arange(nu[k]+1-D,nu[k]+1)*0.5).sum())
    lnf[:,k] += temp
  return lnf

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
  KL = -(lnZ_Dirichlet(alpha1) - lnZ_Dirichlet(alpha2)) + \
      np.dot((alpha1 - alpha2), (digamma(alpha1) - digamma(alpha1.sum())))
  if KL < 0.0 :
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
  D = len(V)
  lnZ = 0.5 * nu * (D * np.log(2.0) - np.log(det(V))) + \
      gammaln(np.arange(nu+1-D,nu+1) * 0.5).sum()
  return lnZ

def E_V_Wishart(nu,V):
  return V / nu

def E_lndetV_Wishart(nu,V):
  """
  mean of log determinant of cov matrix over Wishart <lndet(cov)>
  input
    nu [float] : dof parameter of Wichart distribution
    V [ndarray, shape (D x D)] : base matrix of Wishart distribution
  """
  D = len(V)
  E = -(D*np.log(2.0) - np.log(det(V)) \
      + digamma(np.arange(nu+1-D,nu+1) * 0.5).sum())
  return E

def KL_Wishart(nu1,V1,nu2,V2):
  """
  KL-div of Wishart distribution KL[q(nu1,V1)||p(nu2,V2)]
  """
  D = len(V1)
  KL = ((nu2 - nu1) * E_lndetV_Wishart(nu1,V1) \
      - nu1 * (D - np.trace(np.dot(V2,inv(V1))))) * 0.5 \
      + lnZ_Wishart(nu2,V2) - lnZ_Wishart(nu1,V1)
  if KL < 0.0 :
    raise ValueError, "KL must be larger than 0"
  return KL

def KL_GaussWishart(nu1,V1,beta1,m1,nu2,V2,beta2,m2):
  """
  KL-div of GaussWishart distribution KL[q(nu1,V1,beta1,m1)||p(nu2,V2,beta2,m2)
  """
  D = len(V1)
  KL = KL_Wishart(nu1,V1,nu2,V2) + 0.5 * (D*(np.log(beta1/beta2) + beta2/beta1\
      - 1.0) + beta2 * nu1 * np.dot(m1-m2,solve(V1,(m1-m2))))
  if KL < 0.0 :
    raise ValueError, "KL must be larger than 0"
  return KL
