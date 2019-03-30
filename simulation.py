import numpy as np
from scipy.stats import random_correlation
import math

# For Python > 3.0 Only
# else : from __future__ import division

''' Simulation 1d
# Trajectoire d'un mouvement brownien sur n periodes de taille dt : 
def brownian_path(dt, n):
    gaussian_vector = np.random.normal(0, dt, n)
    gaussian_vector[0] = 0
    return np.cumsum(gaussian_vector)


# Mouvement Brownien Geometrique 1D 
def GBM(So, mu, sigma, W, T, N):
    t = np.linspace(0., 1., N + 1)
    S = []
    S.append(So)
    for i in range(1, int(N + 1)):
        drift = (mu - 0.5 * sigma ** 2) * t[i]
        diffusion = sigma * W[i - 1]
        S_temp = So * np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t
'''

# d = dimension de la matrice, bool = retourne matrice correlations
def var_matrix(d):
    eigen_values = np.random.uniform(low=0, high=10, size=d)
    eigen_values = d * eigen_values / np.sum(eigen_values)
    return random_correlation.rvs(tuple(eigen_values))

''' Tester si une matrice est définie positive
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0
'''

# Brownian motion according to var_matrix on n periods of size dt
def correlated_brownian_paths(d, n, dt, var_matrix):
    gaussian_paths = np.reshape(np.random.normal(0, math.sqrt(dt), d*n),(d,n))
    gaussian_paths[:,0] = 0.
    L = np.linalg.cholesky(var_matrix)
    gaussian_paths = np.dot(L,gaussian_paths)
    gaussian_paths = np.cumsum(gaussian_paths, axis=1)
    return  gaussian_paths

# Geometric correlated brownian motions :
def geometric_brownian_motion(d, n, dt, inital_values, mu, sigma, brownian_paths):
    diffusion = np.multiply(sigma,brownian_paths)
    drift = np.multiply(mu - np.square(sigma)/2., dt*np.arange(0,n,1.))
    return np.multiply(inital_values, np.exp(drift + diffusion))
