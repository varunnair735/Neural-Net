"""
Author: Varun Nair
Date: 6/24/19
"""
import numpy as np
import os

def create_val_set(data):
    """Sorts data into training and validation sets for the main algorithm"""
    lineList = [line.rstrip('\n') for line in open(data)]
    backup = data[:-4] + "_backup.csv"
    with open(backup, 'w') as f:
        for file in lineList:
            f.write("%s\n" % file)
    randomList = np.random.choice(lineList, len(lineList), replace=False)
    os.remove(data)
    with open(data, 'w') as f:
        for file in randomList:
            f.write("%s\n" % file)

def iris_check_val_set(data):
    lineList = [line.rstrip('\n') for line in open(data)]
    setosa_count = 0
    versicolor_count = 0
    virginica_count = 0
    for i, item in enumerate(lineList):
        if i >= 128:
            if 'setosa' in item:
                setosa_count += 1
            elif 'versicolor' in item:
                versicolor_count += 1
            elif 'virginica' in item:
                virginica_count += 1

    print(setosa_count, virginica_count, versicolor_count)

if __name__ == '__main__':
    create_val_set('data/fires/forestfires.csv')
