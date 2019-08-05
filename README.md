# Neural Net

This repository holds data and code for a fully connected neural network that classifies different types of Irises (setosa, versicolor, and virginica).

The neural network itself is two layer (one hidden layer). The hidden layer uses a sigmoid activation function. The hyperparameters that can be adjusted include
- learning rate
- number of epochs
- batch size
- output layer activation function
- type of loss function


The modules

    activation_funtions
    partition
    sorter
are necessary for use. partition and sorter randomize the flower data and split into training and testing sets. activation_funtions define different types of neural net activation functions as well as their derivatives for use in backpropagation.
