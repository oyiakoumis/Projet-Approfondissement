from PortfolioManagement import PortfolioManagement
import simulation as sim
import tensorflow as tf
import numpy as np

# model parameters:
batch_size = 3
n_steps = 10
n_stocks = 5
transaction_costs = 0.2
alpha_parameter = 1.5
learning_rate = 0.001
n_neurons = 100
n_layers = 3

# simulation parameters:
total_time = 30
n_steps = 1000
dt = total_time/n_steps
inital_values = 1 * np.ones((n_stocks,1))
mu = np.full((n_stocks,1),0.01)
sigma = np.full((n_stocks,1),0.1)

portefeuille = PortfolioManagement(batch_size, total_time, n_stocks, transaction_costs,
                                      alpha_parameter, learning_rate, n_neurons)

portefeuille.build()

n_iterations = 10000
batch_size = portefeuille.batch_size

with tf.Session() as sess:
    sess.run(portefeuille.init)
    for iteration in range(n_iterations):
        X_batch, y_batch = sim.next_batch(batch_size, n_stocks,n_steps,total_time,inital_values,mu,sigma)
        _, loss_eval = sess.run([portefeuille.training_op, portefeuille.loss], feed_dict={portefeuille.X: X_batch, portefeuille.y: y_batch})
        if iteration % 100 == 0:
            print(iteration, "\tloss:", loss_eval)