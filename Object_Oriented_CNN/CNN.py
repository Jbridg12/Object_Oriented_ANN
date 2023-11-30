import numpy as np
# from project2 import Neuron

# Layer performing convolutions on the input 
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

        self.outputShape = (self.outputX, self.outputY, numberOfKernels)

        if weights is None:
            # Generate same weights
            weights = np.fill((sizeOfKernels, sizeOfKernels), 0.5)

        self.neurons = np.empty((self.outputX, self.outputY, numberOfKernels))
        for k in range(numberOfKernels):
            for i in range(self.outputX):
                for j in range(self.outputY):
                    self.neurons[i][j][k] = Neuron(activation, sizeOfKernels**2, lr, weights)

    def calculate(self, input):
        # Define output matrix
        out = np.empty((self.outputY, self.outputX, self.numberOfKernels))

        # Loop and for each neuron in all channels (This identifies where the kernel is looking)
        for k in range(self.inputShape[2]):
            # j is row
            for j in range(self.outputY):
                # i is column
                for i in range(self.outputX):
                    # Create a matrix to store input X's for each neuron
                    net = np.empty((self.sizeOfKernels, self.sizeOfKernels))

                    # Then iterate over each relevant element that current kernel is observing
                    # (This locates the specific element from input in the area of each kernel)
                    for x in range(self.sizeOfKernels):
                        for y in range(self.sizeOfKernels):
                            net[y][x] = input[j*self.sizeOfKernels + y][i*self.sizeOfKernels + x][k]
                            
                    # Insert results to the output matrix
                    out[j][i][k] = self.neurons[j][i][k].calculate(net)
        return

    def calculatewdeltas(self, wtimesdelta):
        return

# Pooling layer reducing dimensionality
class MaxPoolingLayer:
    def __init__(self, sizeOfKernel, inputShape):
        """
        Initializes max pooling layer. Assume the stride is always the same as the FILTER SIZE. No padding is needed.
        :param sizeOfKernel: Size of the kernel (assume it is a square).
        :param inputShape: Dimension of the inputs
        """
        self.sizeOfKernel = sizeOfKernel
        self.inputShape = inputShape

        self.outputX = inputShape[0] - sizeOfKernel + 1
        self.outputY = inputShape[1] - sizeOfKernel + 1

        self.outputShape = (self.outputX, self.outputY, inputShape[2])

        # Store coordinates in a matrix for backpropogation
        self.coords = np.empty((self.sizeOfKernel, self.sizeOfKernel, self.inputShape[2]))

    def calculate(self, input):
        # Create output matrix
        out = np.empty((self.sizeOfKernel, self.sizeOfKernel, self.inputShape[2]))

        # Determine the amount of strides needed
        move = self.inputShape / self.sizeOfKernel

        # For each channel
        for k in range(self.inputShape[2]):
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
                                max = input[y+(i*move)][x+(j*move)][k]
                                max_coords = (y+(i*move), x+(j*move), k)

                    out[i][j][k] = max   # Store max in output
                    self.coords[i][j][k] = max_coords  # Store coordinates in 
        return

    def calculatewdeltas(self, wtimesdelta):
        # Create output matrix
        out = np.zeroes(self.inputShape)

        for k in range(self.inputShape[2]):
            for x in range(self.sizeOfKernel):
                for y in range(self.sizeOfKernel):

                    coord = self.coords[x][y][k]
                    out[coord[0]][coord[1]][coord[2]] = wtimesdelta[x][y][k]
        return

# Flatten layer resutructuring output into one dimension
class FlattenLayer:
    def __init__(self, inputSize):
        self.inputSize = inputSize

    def calculate(self, input):
        flat = np.copy(input)

        # If size is given as the shape, calculate size in one dimension
        #flat_size = inputSize[0] * inputSize[1] * inputSize[2]
        
        # If size is already 1D simply reshape
        np.reshape(flat, self.inputSize)
        return flat

    def calculatewdeltas(self, wtimesdelta):
        return
