from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.slim import fully_connected as fc
import random


import glob, os, os.path



def normalizeDataFrame(inputDF):

    x = inputDF.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=inputDF.columns)
    return df, min_max_scaler



class PortfolioManagement(object):


    def __init__(self,  nbReturns, transactionCosts , nbDates, alpha= 0.5,learning_rate=1e-3, batch_size=100, layerSize =[10, 10],
                 restoreGraph = False ):

        self.transactionCosts = transactionCosts
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nbDates = nbDates
        self.alpha = alpha

        self.layerSize = layerSize

        self.nbReturns = nbReturns

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()


        if restoreGraph is False :

            self.build()

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.saver.save(self.sess, 'save/my_test_model', global_step=1000, write_meta_graph=True)
            self.writer = tf.summary.FileWriter('save/my_test_model', graph=tf.get_default_graph())
        else :
            tf.reset_default_graph()
            imported_meta = tf.train.import_meta_graph('save/my_test_model')
            imported_meta.restore(self.sess, tf.train.latest_checkpoint('./save'))


# Build the network and the loss functions
    def build(self):

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.nbReturns, self.nbDates], name='inputs')

        #self.weights = None #tf.Variable(shape = [None, self.nbReturns, self.nbDates])
        cost = []
        ptfReturn = []
        weights = []

        self.cost = None #tf.Variable(shape = [None, self.nbReturns, self.nbDates])
        self.ptfReturn = None #tf.Variable(shape = [None, self.nbReturns, self.nbDates])
        self.weights = None #tf.get_variable( name = 'weight0', shape = self.inputs[:, :, 0].get_shape().as_list(),
                             #           trainable=False, initializer=tf.constant_initializer(1./self.nbReturns))

        prevWeights = 1. / self.nbReturns

        for iDate in range(1,self.nbDates) :
            subInput = self.inputs[:,:, iDate]
            for iLayer in self.layerSize :
                fi = fc(subInput, iLayer, activation_fn=tf.nn.relu)
                subInput = fi

            self.currentWeight = fc(fi, self.nbReturns, activation_fn=tf.nn.softmax)

            self.currentCost = tf.reduce_sum(tf.abs(self.currentWeight[:,:] - prevWeights) \
                                                 * self.transactionCosts , axis = 1)


            # A changer pour un rendement cumulé
            self.currentReturn = tf.reduce_sum(self.inputs[:, :, iDate] * self.currentWeight[:,:], axis = 1)

            weights.append(self.currentWeight)
            cost.append(self.currentCost)
            ptfReturn.append(self.currentReturn)

            '''
            if self.cost is None :
                self.cost = self.currentCost
                self.weights = self.currentWeight
                self.ptfReturn = self.currentReturn
            else :
                self.cost = tf.stack([self.cost, self.currentCost], axis = -1 )
                self.weights = tf.stack([self.weights, self.currentWeight], axis = -1 )
                self.ptfReturn = tf.stack([self.ptfReturn, self.currentReturn], axis = -1 )
            '''

            # Doute ici
            prevWeights = self.currentWeight

        self.ptfReturn = tf.stack(ptfReturn, axis = -1)
        self.weights = tf.stack(weights, axis = -1)
        self.cost = tf.stack(cost, axis = -1)

        mean, var= tf.nn.moments(tf.stack(ptfReturn, axis = -1)[:, -1], axes = -1)
        self.loss = - mean + tf.reduce_mean(tf.stack(cost, axis = -1)[:, -1], axis = -1 ) + self.alpha * var
        self.train = tf.train.AdamOptimizer().minimize(self.loss)
        return



    def run_single_step(self, inputs):


        feed_dict = {self.inputs: inputs}
        train, loss, weights = self.sess.run([self.train, self.loss, self.weights], feed_dict = feed_dict)

  
        return loss

    def predict(self,  inputs):
        feed_dict = {self.inputs: inputs}
        weights = self.sess.run([self.weights], feed_dict=feed_dict)
        return weights



def train(input, transactionCosts , num_epoch, batchSize= 50,  restoreGraph = False ) :

    random.seed(0)

    nbExample, nbReturns, nbDates,  = input.shape


    if restoreGraph is False :

        #couts_variables = np.zeros((batch_size, nbCoutsVariables))
        #prixElec = np.zeros((batch_size))
        #volume = np.zeros((batch_size))
        np.random.seed(0)

        '''
        filelist = glob.glob(os.path.join(os.path.realpath(__file__), "save"))
        for f in filelist:
            os.remove(f)
        '''

        assetManager = PortfolioManagement(nbReturns, transactionCosts , nbDates)

        for epoch in range(num_epoch):

            inputIndex = np.random.randint(0, nbExample, size= batchSize)

            input_in = input[inputIndex[:], :, : ]


            loss  = assetManager.run_single_step(input_in)


            if epoch % 100 == 0:

                print('Epoch : {}, Loss: {}'.format(epoch, loss))
                #model.writer.add_summary(summary, epoch)



