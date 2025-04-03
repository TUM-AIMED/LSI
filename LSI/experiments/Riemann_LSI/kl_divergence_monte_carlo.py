import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

def py_fast_digamma(x):
    "Faster digamma function assumes x > 0."
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + np.log(x) - 0.5/x + t

def l2_distance(list1, list2):
    # Convert lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)
    
    # Expand dimensions to prepare for broadcasting
    array1_expanded = np.expand_dims(array1, axis=1)
    array2_expanded = np.expand_dims(array2, axis=0)
    
    # Compute L2 distance
    distances = np.sqrt(np.sum((array1_expanded - array2_expanded)**2, axis=2))
    
    return distances

# https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf
# equation (26)

def kl_monte_carlo(weights1, weights2):
    distances_self = l2_distance(weights1, weights1)
    distances_other = l2_distance(weights1, weights2)
    # first index is across first entry, second index across second entry
    n = len(weights1)
    m = len(weights2)
    min_self = np.min(distances_self + np.identity(distances_self.shape[0])*1000, axis=0)
    min_other = np.min(distances_other, axis=1)

    radii = 2 * np.max([min_self, min_other], axis=0)
    
    roh = distances_self <= np.array([radii]).transpose()
    nu = distances_other <= np.array([radii]).transpose()

    lis = np.sum(roh, axis=1)
    kis = np.sum(nu, axis=1) 

    digamma = np.vectorize(py_fast_digamma)
    dig_lis = digamma(lis)
    dig_kis = digamma(kis)

    KL_approx = 1/n * (np.sum(dig_lis - dig_kis)) + np.log(m/(n+1))
    return KL_approx


# weight1 = np.array([1, 2, 3, 6])
# weight2 = np.array([2, 2, 2, 5])
# weight3 = np.array([3, 3, 3, 4])
# weight4 = np.array([4, 2, 3, 3])
# weight5 = np.array([5, 2, 2, 2])
# weight6 = np.array([6, 6, 6, 6])

# weights1 = [weight1, weight2, weight3]
# weights2 = [weight4, weight6]

# kl_monte_carlo(weights1=weights1, weights2=weights2)

def kl_divergence(mu1, cov1, mu2, cov2):
    """
    Compute the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
        mu1 (numpy.ndarray): Mean vector of the first Gaussian distribution.
        cov1 (numpy.ndarray): Covariance matrix of the first Gaussian distribution.
        mu2 (numpy.ndarray): Mean vector of the second Gaussian distribution.
        cov2 (numpy.ndarray): Covariance matrix of the second Gaussian distribution.
    
    Returns:
        kl_div (float): KL divergence between the two distributions.
    """
    # Convert to NumPy arrays if not already
    mu1 = np.array(mu1)
    cov1 = np.array(cov1)
    mu2 = np.array(mu2)
    cov2 = np.array(cov2)
    
    # Compute the determinant of covariance matrices
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    
    # Compute the inverse of covariance matrices
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)
    
    # Compute the quadratic terms
    quad_term = np.trace(np.dot(inv_cov2, cov1))
    
    # Compute the difference between means
    diff_means = mu2 - mu1
    
    # Compute the KL divergence
    kl_div = 0.5 * (np.log(det_cov2 / det_cov1) - len(mu1) + quad_term + np.dot(np.dot(diff_means.T, inv_cov2), diff_means))
    
    return kl_div

def skl_efficient(s1, s2, k=1):
    """An efficient version of the scikit-learn estimator by @LoryPack
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)

    Contributed by Lorenzo Pacchiardi (@LoryPack)
    """
    s1 = np.array(s1)
    s2 = np.array(s2)
    n, m = len(s1), len(s2)
    d = float(s1.shape[1])

    s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(s1)
    s2_neighbourhood = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s2)

    s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
    s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
    rho = s1_distances[:, -1]
    nu = s2_distances[:, -1]
    D = np.sum(np.log(nu / rho))

    return (d / n) * D + np.log(
        m / (n - 1)
    )  # this second term should be enough for it to be valid for m \neq n

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def scikit_based_kl(X,Y):
    gm1 = GaussianMixture(n_components=200).fit(X)
    weights1 = gm1.weights_
    means1 = gm1.means_
    cov1 = gm1.covariances_

    gm2 = GaussianMixture(n_components=200).fit(Y)
    weights2 = gm2.weights_
    means2 = gm2.means_
    cov2 = gm2.covariances_
    n_samples = 10e5
    X, _ = gm1.sample(n_samples)
    log_p_X = gm1.score_samples(X)
    log_q_X = gm2.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

def scikit_based_kde_kl(X,Y):
    gm1 = gaussian_kde(np.array(X).transpose())
    gm2 = gaussian_kde(np.array(Y).transpose())
    n_samples = 10e5
    X = gm1.resample(int(n_samples))
    log_p_X = gm1.logpdf(X)
    log_q_X = gm2.logpdf(X)
    return log_p_X.mean() - log_q_X.mean()

from scipy.stats import multivariate_normal

# Define the parameters for the first normal distribution
mean1 = np.array([1,2])
covariance1 = np.array([[1, 0],
                        [0, 2]])

# Define the parameters for the second normal distribution
mean2 = np.array([1, 2.02])
covariance2 = np.array([[1, 0],
                        [0, 2]])

# Create instances of the multivariate normal distributions
dist1 = multivariate_normal(mean1, covariance1)
dist2 = multivariate_normal(mean2, covariance2)

for i in range(2, 10):
    # Compute the KL divergence between the two distributions
    kl_divergence1 = kl_divergence(mean1, covariance1, mean2, covariance2)

    # Store samples from each distribution in lists
    samples_dist1 = [dist1.rvs() for _ in range(10**(i+1))]
    samples_dist2 = [dist2.rvs() for _ in range(10**(i+1))]
    kl_divergence2 = 100 # kl_monte_carlo(samples_dist1, samples_dist2)
    kl_divergence3 = 100 # skl_efficient(samples_dist1, samples_dist2, k=40)
    kl_divergence4 = 100 # KLdivergence(samples_dist1, samples_dist2)
    kl_divergence5 = scikit_based_kl(samples_dist1, samples_dist2)
    kl_divergence6 = scikit_based_kde_kl(samples_dist1, samples_dist2)

    print(f"Samples {10**(i+1)} - KL1: {kl_divergence1}, KL2 er: {abs(kl_divergence2 - kl_divergence1)}, KL3 er: {abs(kl_divergence3 - kl_divergence1)}, KL4 er: {abs(kl_divergence4 -kl_divergence1)}, KL5 er: {abs(kl_divergence5 -kl_divergence1)}, KL6 er: {abs(kl_divergence6 -kl_divergence1)}")
print("")



# https://arxiv.org/abs/1907.00196