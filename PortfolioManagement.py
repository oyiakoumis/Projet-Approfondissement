import tensorflow as tf
import keras

class PortfolioManagement:

    def __init__(self, batch_size, total_time, n_stocks, transaction_costs, alpha_parameter, learning_rate, n_neurons, n_layers):

        self.batch_size = batch_size
        self.total_time = total_time
        self.n_stocks = n_stocks
        self.transaction_costs = transaction_costs
        self.alpha_parameter = alpha_parameter
        self.learning_rate = learning_rate
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.n_inputs = n_stocks
        self.n_outputs = n_stocks

    def build(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.total_time, self.n_inputs], name="Inputs")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.total_time, self.n_outputs], name="Targets")

        self.cell = tf.contrib.rnn.BasicLSTMCell
        self.cell_drop = tf.contrib.rnn.DropoutWrapper
        self.multi_cell = tf.contrib.rnn.MultiRNNCell([self.cell_drop(self.cell(num_units=self.n_neurons,
                            activation=tf.nn.relu), input_keep_prob=self.keep_prob) for _ in range(self.n_layers)])
        self.cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(self.multi_cell, self.n_outputs, activation = tf.nn.softmax)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.cell_wrapped, self.X, dtype=tf.float32)

        self.ptf_mean, self.ptf_var = tf.nn.moments(self.outputs * self.y, axes=2) # element-wise multiplication
        self.loss = tf.reduce_mean(tf.reduce_mean(self.ptf_var - self.alpha_parameter * self.ptf_mean, axis=1) + \
                    self.transaction_costs * tf.reduce_sum(self.outputs[:,-1,:],axis=1), axis=0)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()
