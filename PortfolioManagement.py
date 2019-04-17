import tensorflow as tf


class PortfolioManagement(object):

    def __init__(self, session, n_cells, n_assets, transaction_costs, alpha_parameter, learning_rate, n_neurons):

        self.session = session
        self.n_cells = n_cells
        self.n_assets = n_assets
        self.transaction_costs = transaction_costs
        self.alpha_parameter = alpha_parameter
        self.learning_rate = learning_rate
        self.n_neurons = n_neurons
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.n_inputs = n_assets
        self.n_outputs = n_assets
    pass

    def build(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_cells, self.n_inputs], name="Inputs")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_cells, self.n_outputs], name="Targets")

        self.cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.n_neurons, layer_norm=False, dropout_keep_prob=1.0)
        self.cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(self.cell, self.n_outputs, activation = tf.nn.softmax)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.cell_wrapped, self.X, dtype=tf.float32)

        self.ptf_mean, self.ptf_var = tf.nn.moments(self.outputs * self.y, axes=2) # element-wise multiplication
        self.loss = tf.reduce_mean(tf.reduce_mean(self.ptf_var - self.alpha_parameter * self.ptf_mean, axis=1) + \
                    self.transaction_costs * tf.reduce_sum(self.outputs[:,-1,:],axis=1), axis=0)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        print("Model Built.")
        pass

    def restore(self, restore_path):
        self.saver.restore(self.session, restore_path)
        print("Model restored.")

    def save(self, save_path="./save/my_model.ckpt"):
        self.saver.save(self.session, save_path)
        print("Model saved in path: %s" % save_path)

    def training(self, training_set, n_iterations, keep_prob=1., save_path="./save/my_model.ckpt"):
        self.session.run(self.init)
        for iteration in range(n_iterations):
            X_batch, y_batch = training_set[iteration]
            _, loss_eval = self.session.run([self.training_op, self.loss],
                                    feed_dict={self.X: X_batch, self.y: y_batch, self.keep_prob: keep_prob})
            if iteration % 100 == 0:
                print("iteration:", iteration, "\tloss:", loss_eval)
                self.save(save_path)
        pass

    def prediction(self, X_new):
        return self.session.run(self.outputs, feed_dict={self.X: X_new, self.keep_prob: 1.})