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

        # Calculate size of output 
        self.outputX = inputShape[0] - sizeOfKernels + 1
        self.outputY = inputShape[1] - sizeOfKernels + 1

        # Generate same weights
        weights = np.fill((sizeOfKernels, sizeOfKernels), 0.5)


        self.neurons = np.empty((self.outputX, self.outputY, numberOfKernels))
        for k in range(numberOfKernels):
            for i in range(self.outputX):
                for j in range(self.outputY):
                    self.neurons[i][j][k] = Neuron(activation, sizeOfKernels**2, lr, weights)


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

        # Store coordinates in a matrix for backpropogation
        self.coords = np.empty((self.sizeOfKernel, self.sizeOfKernel))

    def calculate(self, input):
        # Create output matrix
        out = np.empty((self.sizeOfKernel, self.sizeOfKernel))

        # Determine the amount of strides needed
        move = self.inputShape / self.sizeOfKernel

        # Go over each section by row x column
        for i in range(move):
            for j in range(move):

                max = -100  # Arbitrary small value
                max_coords = None 

                # Loop through each element of the smaller sections
                for y in range(self.sizeOfKernel):
                    for x in range(self.sizeOfKernel):

                        # Find the max and store coordinates
                        if input[y+(i*move)][x+(j*move)] > max:
                            max = input[y+(i*move)][x+(j*move)]
                            max_coords = (y+(i*move), x+(j*move))

                out[i][j] = max   # Store max in output
                self.coords[i][j] = max_coords  # Store coordinates in 
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
