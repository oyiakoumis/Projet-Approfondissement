import asset_management_core as MC
import numpy as np

numpy.random.seed(42)

input = np.random.uniform(0, 1, size = (100000, 10, 30))
MC.train(input, transactionCosts =0.02 , num_epoch = 10000)
