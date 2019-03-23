import asset_management_core as MC
import numpy as np

random.seed(10)

# Trajectoire d'un mouvement brownien sur n periodes de taille dt :
def brownian_path(dt=0.1, n):
    gaussian_vector = np.random.normal(0,dt,n)
    gaussian_vector[0]=0
    return np.cumsum(gaussian_vector)

# Mouvement Brownien Geometrique 1D
def GBM(So, mu, sigma, W, T, N):
# Parameters
#
# So:     initial stock price
# mu:     returns (drift coefficient)
# sigma:  volatility (diffusion coefficient)
# W:      brownian motion
# T:      time period
# N:      number of increments
    t = np.linspace(0., 1., N + 1)
    S = []
    S.append(So)
    for i in range(1, int(N + 1)):
        drift = (mu - 0.5 * sigma ** 2) * t[i]
        diffusion = sigma * W[i - 1]
        S_temp = So * np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t


input = np.random.uniform(0, 1, size = (100000, 10, 30))
MC.train(input, transactionCosts =0.02 , num_epoch = 10000)
