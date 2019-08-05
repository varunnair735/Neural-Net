"""
Author: Varun Nair
Date: 6/25/19
"""

import numpy as np
import pandas as pd

def IRIS():
    iris_raw = pd.read_csv('iris.data', header=0)
    iris = iris_raw.SEPLEN, iris_raw.SEPWID,\
            iris_raw.PETLEN, iris_raw.PETWID
    iris = np.array(iris)

    answers = np.zeros((150,3))
    for i, flower in enumerate(iris_raw.CLASS):
        if flower == 'Iris-setosa':
            answers[i,0] = 1
        elif flower == 'Iris-versicolor':
            answers[i,1] = 1
        elif flower == 'Iris-virginica':
            answers[i,2] = 1
    answers = answers.T

    x = int(0.85*150) #training to validating ratio
    X = iris[:,:x]
    Y = answers[:,:x]
    Xtest = iris[:,x:]
    Ytest = answers[:,x:]
    np.savetxt('val_labels.csv', Ytest, delimiter=',')
    return X, Y, Xtest, Ytest

def cleanFire():
    fire_raw = pd.read_csv('forestfires.csv', header=0)
    #transforms area for better regression results
    fireArea = fire_raw.area
    fireArea = np.array(fireArea)
    log_fireArea = np.log(1 + fireArea)
    log_fireArea = log_fireArea.reshape((517,1))

    #transforms months into one hot encoding for seasons
    months = pd.get_dummies(fire_raw.month)
    cols = ['dec','jan', 'feb', 'mar','apr','may','jun','jul','aug','sep','oct','nov']
    months = months[cols]
    seasons = np.zeros([517,4])
    #codes months as seasons
    for i in range(517):
        if months.dec[i:i+1][i] == 1:
            seasons[i,0] = 1
        elif months.jan[i:i+1][i] == 1:
            seasons[i,0] = 1
        elif months.feb[i:i+1][i] == 1:
            seasons[i,0] = 1
        elif months.mar[i:i+1][i] == 1:
            seasons[i,1] = 1
        elif months.apr[i:i+1][i] == 1:
            seasons[i,1] = 1
        elif months.may[i:i+1][i] == 1:
            seasons[i,1] = 1
        elif months.jun[i:i+1][i] == 1:
            seasons[i,2] = 1
        elif months.jul[i:i+1][i] == 1:
            seasons[i,2] = 1
        elif months.aug[i:i+1][i] == 1:
            seasons[i,2] = 1
        elif months.sep[i:i+1][i] == 1:
            seasons[i,3] = 1
        elif months.oct[i:i+1][i] == 1:
            seasons[i,3] = 1
        elif months.nov[i:i+1][i] == 1:
            seasons[i,3] = 1

    #puts features into single array, ignoring day of week
    position = fire_raw.X, fire_raw.Y
    meteor = fire_raw.FFMC, fire_raw.DMC, fire_raw.DC, fire_raw.ISI,\
                fire_raw.temp, fire_raw.RH, fire_raw.wind, fire_raw.rain
    position = np.array(position).T
    meteor = np.array(meteor).T

    data = np.concatenate((position, seasons, meteor, log_fireArea),axis=1)
    np.savetxt('fires_cleaned.csv', data, delimiter=',')

def fire():
    data = np.loadtxt('fires_cleaned.csv', delimiter=',')
    training, test = data[:int(0.85*517),:], data[int(0.85*517):,:]
    X, Y = training[:,:-1], training[:,-1:]
    X, Y = X.T, Y.T
    Xtest, Ytest = test[:,:-1], test[:,-1:]
    Xtest, Ytest = Xtest.T, Ytest.T
    np.savetxt("x.csv",X,delimiter=',')
    return X, Y, Xtest, Ytest
