from project2 import Neuron


class ConvolutionalLayer:
    def __init__(self, numberOfKernels, sizeOfKernels, activation, inputShape, lr, weights=None):
        """
        Initializes convolutional layer for 2d convolution. Stride is always assumed to be 1. Padding is assumed to
        be valid.
        :param numberOfKernels: Number of kernels in layer.
        :param sizeOfKernels: Size of kernels in layer (assume it is a square).
        :param activation: Activation function for all neurons in the layer.
        :param inputShape: Dimension of the inputs.
        :param lr: Learning rate.
        :param weights: If no weights provided, randomly initialize.
        """
        # Each neuron should be a Neuron object.
        # All neurons with the same kernel share the same weights, so MUST BE INITIALIZED WITH THE SAME WEIGHTS
        self.numberOfKernels = numberOfKernels
        self.sizeOfKernels = sizeOfKernels
        self.activation = activation
        self.inputShape = inputShape
        self.lr = lr
        self.weights = weights


    def calculate(self):
        return

    def calculatewdeltas(self, wtimesdelta):
        return


class MaxPoolingLayer:
    def __init__(self, sizeOfKernel, inputShape):
        """
        Initializes max pooling layer. Assume the stride is always the same as the FILTER SIZE. No padding is needed.
        :param sizeOfKernel: Size of the kernel (assume it is a square).
        :param inputShape: Dimension of the inputs
        """
        self.sizeOfKernel = sizeOfKernel
        self.inputShape = inputShape

    def calculate(self):
        return

    def calculatewdeltas(self, wtimesdelta):
        return


class FlattenLayer:
    def __init__(self, inputSize):
        self.inputSize = inputSize

    def calculate(self):
        return

    def calculatewdeltas(self, wtimesdelta):
        return
