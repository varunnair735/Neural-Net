"""
Author: Varun Nair
Date: 6/30/19
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import activation_functions as af
from partition import iris_partition
from partition import IRIS, fire

training_set, training_labels, test_set, test_labels = IRIS()

class Vnet:
    """Initializes fully connected neural network and has methods to train
        and test network including plots of loss function"""
    def __init__(self, x, y, N, xTest, yTest, learn_rate, epoch, batch_size,
                output_activation='sigmoid', loss='L2'):
        """@param: x is training data
        @param: y is labels for training data
        @param: N is number of neurons in hidden layer
        @param: xTest is data for testing
        @param: yTest is labels for testing
        @param: learn_rate and epoch are as usual
        @param: output_activation can be either 'sigmoid', 'ReLU', or 'reg'
        @param: loss can be 'L2' or 'CE' for cross entropy"""
        np.random.seed(14)

        #structure of network for training
        self.x, self.y = x.T, y.T
        self.BS = batch_size

        #models 1,2,3
        #self.w1 = 2*np.random.random((x.shape[0], N)) - 1
        #self.w2 = 2*np.random.random((N, 3)) - 1

        #model 4,5,6
        #self.w1 = np.random.randn(x.shape[0], N) * np.sqrt(2 / x.shape[0])
        #self.w2 = np.random.randn(N, 3) * np.sqrt(2 / N)

        #model 7,8
        self.w1 = np.random.randn(x.shape[0], N)
        self.w2 = np.random.randn(N, 3)

        self.b1, self.b2 = 0, 0
        #self.b1, self.b2 = np.zeros([1,N]), np.zeros([1, self.y.shape[1]])

        #hyperparameters
        self.LR = learn_rate
        self.epoch = epoch
        self.activation = output_activation
        self.loss_type = loss

        #for testing
        self.xTest = xTest.T
        self.yTest = yTest.T

        #for saving graphs and predictions
        self.fileName = 'N{}_BS{}_LR{:1.0e}_EP{}_{}_{}'.format(N,
                                                    self.BS,
                                                    self.LR,
                                                    self.epoch,
                                                    self.activation,
                                                    self.loss_type)

    def forward(self):
        """Calculates predicted values"""
        #hidden layer
        """self.hidden = af.sig((self.x @ self.w1) + self.b1)"""
        self.hidden = af.sig((self.xbatch @ self.w1) + self.b1)
        #output layer
        if self.activation == 'sigmoid':
            self.output = af.sig((self.hidden @ self.w2) + self.b2)
        elif self.activation == 'reg':
            self.output = (self.hidden @ self.w2) + self.b2
        elif self.activation == 'ReLU':
            self.output = af.RELU((self.hidden @ self.w2) + self.b2)

    def L2(self):
        """Defines loss function as sum of squared errors"""
        return np.sum((self.output - self.ybatch) **2)

    def crossEntropy(self):
        """Defines a class-based Cross-Entropy loss function"""
        return -(np.sum(self.ybatch @ np.log(self.output)))

    def backprop(self):
        """Adjusts weighting through gradient descent and chain rule"""
        if self.activation == 'sigmoid' and self.loss_type == 'L2':
            #db2 = 1 * af.sig_prime(self.output)
            dw2 = self.hidden.T @ (
                                    (self.ybatch - self.output)
                                    * af.sig_prime(self.output)
                                    )
        elif self.activation == 'reg' and self.loss_type == 'L2':
            dw2 = self.hidden.T @ (
                                    (self.ybatch - self.output)
                                    * (self.output)
                                    )
        elif self.activation == 'ReLU' and self.loss_type == 'L2':
            #db2 = 1 * af.RELU_prime(self.output)
            dw2 = self.hidden.T @ (
                                    (self.ybatch - self.output)
                                    * af.RELU_prime(self.output)
                                    )
        elif self.activation == 'sigmoid' and self.loss_type == 'CE':
            pass
        elif self.activation == 'ReLU' and self.loss_type == 'CE':
            pass
        self.w2 += self.LR * dw2
        #self.b2 += self.LR * db2

        #db1 = dw2 * af.sig_prime(self.hidden)
        dw1 = self.xbatch.T @ (
                            (
                                (
                                    (self.ybatch - self.output)
                                    * af.sig_prime(self.output)
                                )
                                @ self.w2.T
                            )
                        * af.sig_prime(self.hidden)
                        )
        self.w1 += self.LR * dw1
        #self.b1 += self.LR * db1

    def train(self):
        """Trains algorithm with correct weights"""
        steps = int(self.epoch * 1000)
        accuracy = np.empty(steps)
        losses = np.empty(steps)

        for i in range(steps):
            #creating batches of data w/ labels for this iteration
            seed = np.random.randint(1e4)
            np.random.seed(seed)
            rows = np.random.choice(self.x.shape[0], self.BS, replace=False)
            self.xbatch = self.x[rows, :]
            self.ybatch = self.y[rows, :]

            self.forward()

            if self.loss_type == 'CE':
                losses[i] = self.crossEntropy()
            elif self.loss_type == 'L2':
                losses[i] = self.L2()
            acc = (1 - np.mean(np.abs(self.ybatch - self.output))) * 100
            accuracy[i] = (acc)
            if i % 2000 == 0:
                #print("Loss at epoch {}: {:4.8f}".format(i,losses[i]))
                pass

            self.backprop()

        print("Training accuracy is {:4.3f}%.".format(acc))

        #plotting results
        x = np.arange(0, steps)
        fig1, ax1 = plt.subplots(2, 1, figsize = (8, 5))
        ax1[0].plot(x, losses,'g-', label='Loss',)
        ax1[1].plot(x, accuracy, 'c-', label='Accuracy')
        ax1[0].set_title('Training Progress')
        ax1[1].set_xlabel('Epoch')
        ax1[0].set_ylabel('Loss')
        ax1[1].set_ylabel('Training Accuracy (%)')
        ax1[1].set_ylim(50,100)
        plt.savefig('irisResults/___training_graph_{}.png'\
                .format(self.fileName), dpi=300)

        self.irisTest(save=True)
        #return self.w1, self.w2, self.b1, self.b2

    def irisTest(self, save=False):
        """Uses weights to make predictions on withheld data for iris"""
        layer1 = af.sig((self.xTest @ self.w1) + self.b1)

        if self.activation == 'sigmoid':
            self.pred = af.sig((layer1 @ self.w2) + self.b2)
        elif self.activation == 'ReLU':
            self.pred = af.RELU((layer1 @ self.w2) + self.b2)

        acc = (1 - np.mean(np.abs(self.yTest - self.pred))) * 100
        print("Test accuracy is {:4.3f}%.".format(acc))

        locations = np.argmax(self.yTest, axis=1)
        predictions = np.argmax(self.pred,axis=1)
        count = 0
        for i in range(len(predictions)):
            if locations[i] == predictions[i]:
                count += 1
        print("Predicted {} of {} flowers correctly."\
                    .format(count, len(predictions)))

        if save == True:
            np.savetxt('irisResults/___predictions_{}.csv'\
                    .format(self.fileName), self.pred, delimiter=',')

    def fireTest(self,save=False):
        """Uses weights to make predictions on withheld data for iris"""
        layer1 = af.sig((self.xTest @ self.w1) + self.b1)

        if self.activation == 'sigmoid':
            self.pred = af.sig((layer1 @ self.w2) + self.b2)
        elif self.activation == 'reg':
            self.pred = (layer1 @ self.w2) + self.b2
        elif self.activation == 'ReLU':
            self.pred = af.RELU((layer1 @ self.w2) + self.b2)

        if save == True:
            np.savetxt('fireResults/predictions_{}.csv'\
                    .format(self.fileName), self.pred, delimiter=',')

irisModel = Vnet(training_set, training_labels, 8, test_set, test_labels,
            learn_rate=4e-3, epoch=10, batch_size=10,
            output_activation='sigmoid')
irisModel.train()
