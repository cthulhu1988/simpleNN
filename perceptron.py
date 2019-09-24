#!/usr/bin/env python3
import numpy as np
def main():


    trainInputs =  np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

    trainOut = np.array([[0,1,1,0]]).T

    np.random.seed(1)

    synaptWeights = 2 * np.random.random((3,1))-1

    print("random synaptWeights")
    print(synaptWeights)

    for interation in range(100000):
        input_layer = trainInputs

        outputsSquished = sigmoid(np.dot(input_layer, synaptWeights))

        error = trainOut - outputsSquished

        adjustments = error * sigmoidDerive(outputsSquished)

        synaptWeights += np.dot(input_layer.T, adjustments)

    print("After Training")
    print(synaptWeights)
    print("outputs")
    print(outputsSquished)

def sigmoidDerive(x):
    return x * (1-x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    main()
