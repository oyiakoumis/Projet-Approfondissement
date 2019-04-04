from __future__ import division
import numpy as np
from scipy.stats import random_correlation
import math


# Attention shape inversée pour les 3 premières fonctions
def covariance_matrix(n_stocks):
    eigen_values = np.random.uniform(low=0, high=10, size=n_stocks)
    eigen_values = n_stocks * eigen_values / np.sum(eigen_values)
    return random_correlation.rvs(tuple(eigen_values))


def correlated_brownian_paths(n_brownians, n_steps, total_time, var_matrix):
    gaussian_paths = np.reshape(np.random.normal(0, math.sqrt(total_time/n_steps), n_brownians * n_steps), (n_brownians, n_steps))
    gaussian_paths[:, 0] = 0.
    L = np.linalg.cholesky(var_matrix)
    gaussian_paths = np.dot(L, gaussian_paths)
    gaussian_paths = np.cumsum(gaussian_paths, axis=1)
    return gaussian_paths


def geometric_brownian_motions(n_steps, total_time, initial_values, mu, sigma, brownian_paths):
    diffusion = np.multiply(sigma, brownian_paths)
    drift = np.multiply(mu - np.square(sigma) / 2., (total_time/n_steps) * np.arange(0, n_steps, 1.))
    return np.multiply(initial_values, np.exp(drift + diffusion))


def stock_prices_simulation(n_brownians, n_steps, total_time, initial_values, mu, sigma):
    correlations = covariance_matrix(n_brownians)
    brownian_paths = correlated_brownian_paths(n_brownians, n_steps, total_time, correlations)
    return np.transpose(geometric_brownian_motions(n_steps, total_time, initial_values, mu, sigma, brownian_paths))


def returns(stock_prices):
    return (stock_prices[1:,:]-stock_prices[:-1,:])/stock_prices[:-1,:]


def one_simulation(n_brownians, n_steps, total_time, initial_values, mu, sigma):
    stocks = stock_prices_simulation(n_brownians, n_steps, total_time, initial_values, mu, sigma)
    stock_returns = returns(stocks)
    stock_returns_daily = stock_returns[range(0,n_steps,(n_steps//total_time)),:]
    return stock_returns_daily[:-1,:], stock_returns_daily[1:,:]


def next_batch(batch_size, n_stocks,n_steps,total_time,inital_values,mu,sigma):
    batch = np.array([one_simulation(n_stocks,n_steps,total_time,inital_values,mu,sigma)
                  for _ in range(batch_size)])
    return batch[:,0,:], batch[:,1,:]

