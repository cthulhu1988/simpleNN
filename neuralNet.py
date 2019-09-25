#!/usr/bin/env python3
import numpy as np

class NeuralNet():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 *np.random.random((3,1)) -1

    def sigmoidDerive(self,x):
        return x * (1-x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, training_inputs, training_outputs, training_iterations):

        for interation in range(training_iterations):
             output = self.think(training_inputs)
             error  = training_outputs - output

             adjustments = np.dot(training_inputs.T, error * self.sigmoid(output))
             self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        #print("inputs in think: {}".format(inputs))

        thinkOutputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        print("think outputs in think self.sigmoid np dot inputs and weight:")
        print(thinkOutputs)
        return thinkOutputs


def main():

    newNN = NeuralNet()
    print("synaptic weights")
    print(newNN.synaptic_weights)


    trainInputs =  np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    trainOut = np.array([[0,1,1,0]]).T

    newNN.train(trainInputs, trainOut, 40)

    print("synaptic weights")
    print(newNN.synaptic_weights)

    # A = str(input("INput 1: "))
    # B = str(input("INput 2: "))
    # C = str(input("INput 3: "))
    #
    # print("INput Data can be A {}, B {} or C {}".format(A,B,C))
    # print("NN outputs")
    # print(newNN.think(np.array([A,B,C])))





if __name__ == '__main__':
    main()
