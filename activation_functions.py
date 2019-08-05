"""
Author: Varun Nair
Date: 6/25/19
"""
import numpy as np

def sig(x):
    """Defines sigmoid activation function"""
    return (1 / (1+ np.exp(-x)))

def sig_prime(x):
    """Defines the derivative of the sigmoid function used in backprop"""
    return sig(x)*(1-sig(x))

def RELU(x):
    """Defines ReLU activation function"""
    return x*(x>0)

def RELU_prime(x):
    """Derivative is 0 when <0 and 1 when >0"""
    return 1*(x>0)

def softmax(x):
    """Computes softmax score for each class so probabilities add to 1.
        This is adjusted by the max value to avoid overflows"""
    a = np.exp(x - np.max(x))
    return a / (np.sum(a, axis=1))

def softmax_prime(x):
    """Derivative of softmax function for use in backprop"""
    pass
