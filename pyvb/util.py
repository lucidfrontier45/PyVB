import numpy as np
from scipy.special import gammaln
from scipy.linalg import  eigh
from scikits.learn.mixture import logsum, normalize
import pylab

def cnormalize(X):
  """
  Z transformation
  """
  return (X - np.mean(X,0)) / np.std(X,0)

def correct_k(k,m):
  """
  Poisson prior for P(Model)
  input
    k [int] : number of clusters
    m [int] : poisson parameter
  output
    log-likelihood
  """
  return k * np.log(m) - m - 2.0 * gammaln(k+1)

def complexity_GMM(k,d):
  """
  count number of parameters for Gaussian Mixture Model in d-dimension
  with k clusters.
  input
    k [int] : number of clusters
    d [int] : dimension of data
  """
  return k * (1.0 * 0.5 * d * (d + 3.0))

def ica(X):
  """
  Wrapper function for Independent Component Analysis of scikits.learn.decomposition
  """
  from scikits.learn.decomposition import FastICA
  ica = FastICA()
  ica.fit(X.T)
  Y = ica.transform(X.T).T
  return Y / Y.std(0)


def posteriorPCA(x,z,npcs=5):
  """
  Weighted Principal Component Analysis with Posterior Probability
  input
    x [ndarray, shape (n x dim)] : observed data
    z [ndarray, shape (n x nmix)] : posterior probability
    npc [int] : number of principal components to be returned
  output
    eig_val [ndarray, shape (npcs)] : eigenvalues
    eig_vec [ndarray, shape (npcs x dim)] : eigenvectors
    PC [ndarray, shape (n x npcs)] : principal components
  """
  N = z.sum() # observed number of this cluster
  xbar = np.dot(z,x) / N # weighted mean vector
  dx = x - xbar
  C = np.dot((z*dx.T),dx) / (N-1) # weighted covariance matrix
  eig_val, eig_vec = eigh(-C,eigvals=(1,npcs)) # eigen decomposition
  PC = np.dot(dx,eig_vec)
  return -eig_val, eig_vec, PC

def similarity_of_hidden_states(z):
    return np.dot(z.T,z) / z.sum(0)[:,np.newaxis]

def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    pylab.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if pylab.isinteractive():
        pylab.ioff()
    pylab.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    pylab.fill(np.array([0,width,width,0]),np.array([0,0,height,height]),'gray')
    pylab.axis('off')
    pylab.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        pylab.ion()
    pylab.show()
