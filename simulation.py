from __future__ import division
import numpy as np
from scipy.stats import random_correlation
from sklearn.preprocessing import StandardScaler
import math
from random import randint


# la convention est (n_days, n_asset)
# Attention shape inversée pour les 3 premières fonctions
def covariance_matrix(n_stocks):
    eigen_values = np.random.uniform(low=0, high=10, size=n_stocks)
    eigen_values = n_stocks * eigen_values / np.sum(eigen_values)
    return random_correlation.rvs(tuple(eigen_values))


def correlated_brownian_paths(n_brownians, n_daily_steps, n_days, var_matrix):
    gaussian_paths = np.reshape(np.random.normal(0, math.sqrt(1/n_daily_steps), n_brownians * n_daily_steps * n_days),
                                (n_brownians, n_daily_steps * n_days))
    gaussian_paths[:, 0] = 0.
    L = np.linalg.cholesky(var_matrix)
    gaussian_paths = np.dot(L, gaussian_paths)
    gaussian_paths = np.cumsum(gaussian_paths, axis=1)
    return gaussian_paths

# modifier 1/365
def geometric_brownian_motions(n_daily_steps, n_days, initial_values, mu, sigma, brownian_paths):
    diffusion = np.multiply(sigma, brownian_paths)
    drift = np.multiply(mu - np.square(sigma) / 2., (1/n_daily_steps) * np.arange(0, n_daily_steps * n_days, 1.))
    return np.multiply(initial_values, np.exp(drift + diffusion))


def raw_data_simulation(n_assets, n_daily_steps, n_days, initial_values, mu, sigma):
    correlations = covariance_matrix(n_assets)
    brownian_paths = correlated_brownian_paths(n_assets, n_daily_steps, n_days, correlations)
    return np.transpose(geometric_brownian_motions(n_daily_steps, n_days, initial_values, mu, sigma, brownian_paths))


def cleaned_data(data_simulation, n_daily_steps, n_days):
    return data_simulation[range(0, n_daily_steps * n_days, n_daily_steps), :]


def get_returns(assets_prices):
    return (assets_prices[1:]-assets_prices[:-1])/assets_prices[:-1]


def scale_data(data_series, fitting_parameter=0.2):
    n_days, n_assets = data_series.shape
    scaler = StandardScaler()
    fitting_data = data_series[:int(n_days * fitting_parameter)]
    scaler.fit(fitting_data)
    print('Mean:', scaler.mean_)
    print('StandardDeviation:', np.sqrt(scaler.var_))
    scaled_data = scaler.transform(data_series)
    return scaled_data, scaler


def inverse_scaling(data_scaled, scaler):
    return scaler.inverse_transform(data_scaled)


def train_test_split(data_set, test_size=0.2):
    train_data = data_set[:int(len(data_set)*(1-test_size))]
    test_data = data_set[int(len(data_set) * (1 - test_size)):]
    return train_data, test_data


def next_batch(train_data, batch_size, n_cells):
    batch = np.array([])
    for _ in range(batch_size):
        start = randint(0,len(train_data) - n_cells)
        batch = np.append(batch, train_data[start:start+n_cells])
    return batch


def set_batch_variables(batch):
    np.transpose(batch, axes=[1,0,2])
    X_batch = np.transpose(batch[:-1,:,:], axes=[1, 0, 2])
    y_batch = np.transpose(batch[1:,:,:], axes=[1,0,2])
    return X_batch, y_batch