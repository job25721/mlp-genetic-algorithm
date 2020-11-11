import numpy as np


def sigmoid(x):
    return 1 / (1+np.e**(-x))


class Node:
    def __init__(self):
        self.input = []
        self.y = []
        self.w = []
        self.b = 0.0

    def initialWeight(self, n):
        self.w = np.array(np.random.uniform(-1, 1, size=n)).reshape((n, 1))

    def addInput(self, inputVector):
        self.input = inputVector

    def updateWeight(self, w, i):
        self.w[i] = w

    def calculateOutput(self, net):
        if len(net) != 0 and len(self.w) != 0:
            self.y = sigmoid(np.array(net).dot(self.w)[0] + self.b)
        else:
            self.y = self.input


class Model:
    def __init__(self):
        self.chromosome = []
        self.layers = []

    def create(self, inputLayer, hiddenLayers, outputLayer):
        self.inputLayer = inputLayer
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer

        self.layers.append(inputLayer)
        for layer in self.hiddenLayers:
            self.layers.append(layer)

        self.layers.append(outputLayer)

        for node in self.outputLayer:
            node.initialWeight(
                n=len(self.hiddenLayers[len(self.hiddenLayers) - 1]))
            for w in node.w:
                self.chromosome.append(w)

        for l, layer in enumerate(self.hiddenLayers):
            for node in layer:
                if l == 0:
                    n = len(self.inputLayer)
                else:
                    n = len(self.hiddenLayers[l-1])
                node.initialWeight(n)
                for w in node.w:
                    self.chromosome.append(w)
        self.chromosome = np.array(self.chromosome)

    def updateNeuralNetwork(self):
        count = 0
        for node in self.outputLayer:
            for i in range(len(node.w)):
                node.w[i] = self.chromosome[count]
                count = count+1
        for layer in self.hiddenLayers:
            for node in layer:
                for i in range(len(node.w)):
                    node.w[i] = self.chromosome[count]
                    count = count+1

    def feed_forward(self, inputVector):
        for l, layer in enumerate(self.layers):
            for i, node in enumerate(layer):
                if l == 0:
                    node.addInput(inputVector[i])
                    node.calculateOutput([])
                else:
                    net = [prev.y for prev in self.layers[l-1]]
                    node.calculateOutput(net)

    def evaluate(self, data):
        SSE = []
        for train in data:
            self.feed_forward(inputVector=train['input'])
            sum_err = []
            for i, node in enumerate(self.outputLayer):
                sum_err.append(
                    np.power((node.y - train['desire_output'][i]), 2))
            SSE.append(np.sum(sum_err))
        return 1/(1+np.average(SSE))
