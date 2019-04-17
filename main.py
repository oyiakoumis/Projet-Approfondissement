from PortfolioManagement import PortfolioManagement
import simulation as sim
import tensorflow as tf
import numpy as np
import pandas as pd


data_frame = pd.read_csv("/Users/odysseas/Desktop/EDF_Amundi/DATA/SP500 Comp-Ltd/Stocks.csv")
np.random.seed(10)

def random_stocks(data_frame, n_stocks):
    stocks_names = data_frame['GVKEY'].unique()
    stocks_name_sample = np.random.choice(stocks_names, n_stocks)
    return data_frame.loc[data_frame['GVKEY'].isin(stocks_name_sample)]

# DATA
n_stocks = 10
stocks_sample = random_stocks(data_frame, n_stocks)

# Graph parameters:
n_cells = 365
transaction_costs = 0.2
alpha_parameter = 1.5
learning_rate = 0.001
n_neurons = 10

# Juste pour tester si le code marche :
training_set = np.array([[np.random.randn(2, n_cells, n_stocks), np.random.randn(2, n_cells, n_stocks)]])

session = tf.Session()
portefeuille = PortfolioManagement(session, n_cells, n_stocks, transaction_costs, alpha_parameter, learning_rate, n_neurons)
portefeuille.build()
portefeuille.training(training_set, n_iterations=1, save_path="./save/my_model.ckpt")
session.close()