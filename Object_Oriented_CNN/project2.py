import numpy as np
import sys
import random   # Only necessary if random initialized weights
import matplotlib.pyplot as plt
# import CNN
# from CNN import ConvolutionalLayer
# from CNN import MaxPoolingLayer
# from CNN import FlattenLayer

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

# Global to indicate whether to display weights for example
SHOW_WEIGHTS = False
EPOCHS = 3000
losses = ['SumOfSquares', 'BinaryCrossEnt']
activations = ['Linear', 'Sigmoid']
LINEAR = 0
SIGMOID = 1
MSE = 0
BINARY = 1

# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self, activation, input_num, lr, weights=None, bias=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr

        self.d = 0
        self.input = None   
        self.output = None  

        # At the individual Neuron level if no weights specified
        # initialize the weights to random floating point values
        # in range(0.0 - 1.0)
        if weights is not None:
            self.in_channels = weights.shape[0]

        self.weights = np.array([channel for channel in weights])
        self.bias = bias

    # This method returns the activation of the net
    def activate(self, net):
        if self.activation == 0:    # Linear Activation
            # ∂(x) = x
            return net
        elif self.activation == 1:                   # Logistic Activation
            # ∂(x) = 1/(1+e^-x)
            exp = np.exp(-net)
            return float(1/(1+float(exp)))

    # Calculate the output of the neuron should save the input and output for back-propagation.
    def calculate(self, input):
        """
        1. multiply i*w
        2. add i*w + i*w
        3. apply activation function

       
        :param input:
        :return:
        """
        if SHOW_WEIGHTS is True:
            print("Weights: {}".format(self.weights))

        self.input = list(input)

        # Append 1 for bias
        #self.input.append(1)

        net = 0
        for i in range(self.in_channels):
            mul = np.multiply(self.input[i], self.weights[i])
            net += np.sum(mul)

        net += self.bias
        self.output = self.activate(net)
        return self.output

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        if self.activation == 0:
            return 1
        elif self.activation == 1:
            return self.output * (1 - self.output)

    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta, mode='FullyConnected'):
        new_wd = []

        curr_delta = wtimesdelta * self.activationderivative()
        self.d = curr_delta

        if mode == 'Convolution' or mode == 'convolution':
            new_wd = np.empty((self.in_channels, self.weights.shape[1], self.weights.shape[2]))
            for ch in range(self.in_channels):
                for row in range(self.weights.shape[1]):
                    for col in range(self.weights.shape[2]):
                       new_wd[ch][row][col] = self.d * self.weights[ch][row][col]
        else:
            new_wd = np.empty((self.input_num))

            for i in range(self.input_num):
                    new_wd[i] = self.weights[0][i] * curr_delta

        return new_wd
    
    # Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self, mode='FullyConnected'):

        '''
        for w in self.weights:
            for i in range(self.input_num):
                w[i] -= self.lr * self.d * self.input[i]
        '''
        if mode == 'Convolution' or mode == 'convolution':
            out = np.empty((self.in_channels, self.weights.shape[1], self.weights.shape[2]))
            b_out = self.lr * self.d

            for ch in range(self.in_channels):
                for row in range(self.weights.shape[1]):
                    for col in range(self.weights.shape[2]):
                       out[ch][row][col] = self.lr * self.d * self.input[ch][row][col]
            return out, b_out
        else:
            for i in range(self.input_num):
                self.weights[0][i] -= self.lr * self.d * self.input[0][i]
            #print(self.bias)
            self.bias -= self.lr * self.d

        return


# A fully connected layer
class FullyConnected:
    # initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None, bias=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights

        self.input = []
        self.output = []

        # Instantiate all Neurons in the given layer
        self.neurons = []
        for i in range(numOfNeurons):
            if weights is None:
                self.neurons.append(Neuron(activation, input_num, lr, weights))
            else:
                self.neurons.append(Neuron(activation, input_num, lr, weights[i], bias))

    # calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)
    def calculate(self, input):
        self.input = input
        self.output = []    # Reset array for ouput since it was infinitely expanding
        if self.input_num is None:
            self.input_num = len(input)

        # Loop through each neuron and pass all Inputs 
        for neuron in self.neurons:
            neuron.input_num = self.input_num
            neuron.calculate(self.input)
            self.output.append(neuron.output)

        return self.output  # Send outputs back to neuralnetwork

    # given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.
    def calcwdeltas(self, wtimesdelta):
        sum_wdelta = []

        for i in range(self.numOfNeurons):
            new_wd = self.neurons[i].calcpartialderivative(wtimesdelta[i])  # Get each neuron's new wtimesdelta, not quite sure if the whole vector needs to be passed but it looks like it
            self.neurons[i].updateweight()

            if i == 0:
                sum_wdelta = new_wd                     # If the vector doesnt exist yet, give it the first values
            else:
                sum_wdelta = np.add(sum_wdelta, new_wd) # Otherwise add vectors and keep a running sum

        print(f'\nfully connected weights:')
        print(self.neurons[0].weights)

        print(f'\nfully connected bias')
        print(self.neurons[0].bias)

        return sum_wdelta

class MaxPoolingLayer:
    def __init__(self, sizeOfKernel, inputShape):
        """
        Initializes max pooling layer. Assume the stride is always the same as the FILTER SIZE. No padding is needed.
        :param sizeOfKernel: Size of the kernel (assume it is a square).
        :param inputShape: Dimension of the inputs
        """
        self.sizeOfKernel = sizeOfKernel
        self.inputShape = inputShape

        self.outputX = int(((inputShape[0] - sizeOfKernel) / sizeOfKernel) + 1)
        self.outputY = int(((inputShape[1] - sizeOfKernel) / sizeOfKernel) + 1)

        self.outputShape = (self.outputX, self.outputY, inputShape[2])
        self.numberOfNeurons = self.outputX * self.outputY

        #print(self.inputShape)
        # Store coordinates in a matrix for backpropogation
        self.coord0 = np.empty((self.inputShape[2], self.inputShape[1], self.inputShape[0]))
        self.coord1 = np.empty((self.inputShape[2], self.inputShape[1], self.inputShape[0]))
        self.coord2 = np.empty((self.inputShape[2], self.inputShape[1], self.inputShape[0]))

    def calculate(self, input):
        # Create output matrix
        out = np.empty((self.inputShape[2], self.outputY, self.outputX))

        # Determine the amount of strides needed
        move = int(self.inputShape[0] / self.sizeOfKernel)

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
                            if input[k][y+(i*self.sizeOfKernel)][x+(j*self.sizeOfKernel)] > max:
                                 max = input[k][y+(i*self.sizeOfKernel)][x+(j*self.sizeOfKernel)]
                                 max_coords = np.array([k, y+(i*self.sizeOfKernel), x+(j*self.sizeOfKernel)])

                    out[k][i][j] = max   # Store max in output
                    self.coord0[k][i][j] = int(max_coords[0])
                    self.coord1[k][i][j] = int(max_coords[1])
                    self.coord2[k][i][j] = int(max_coords[2])

        return out

    def calcwdeltas(self, wtimesdelta):
        # Create output matrix
        out = np.zeros((self.inputShape[2], self.inputShape[0], self.inputShape[0]))
        move = int(self.inputShape[0] / self.sizeOfKernel)

        for k in range(self.inputShape[2]):
            for j in range(move):
                for i in range(move):
                    for y in range(self.sizeOfKernel):
                        for x in range(self.sizeOfKernel):
                            coord = (self.coord0[k][j][i], self.coord1[k][j][i], self.coord2[k][j][i])
                            out[int(coord[0])][int(coord[1])][int(coord[2])] = wtimesdelta[k][j][i]
        return out

class ConvolutionalLayer:
    def __init__(self, numberOfKernels, sizeOfKernels, activation, inputShape, lr, weights=None, biases=None):
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
        self.numberOfNeurons = self.outputX * self.outputY * numberOfKernels

        if weights is None:
            # Generate same weights
            weights = np.fill((sizeOfKernels, sizeOfKernels), 0.5)

        self.neurons = np.empty((numberOfKernels, self.outputX, self.outputY), dtype=Neuron)
        for k in range(numberOfKernels):
            for i in range(self.outputX):
                for j in range(self.outputY):
                    self.neurons[k][i][j] = Neuron(activation, sizeOfKernels**2, lr, weights[k], biases[k])

    def calculate(self, input):
        # Define output matrix
        out = np.zeros((self.numberOfKernels, self.outputY, self.outputX), dtype=Neuron)

        # Loop and for each neuron in all channels (This identifies where the kernel is looking)
        for z in range(self.numberOfKernels):
            # j is row
            for j in range(self.outputY):
                # i is column
                for i in range(self.outputX):
                    # Create a matrix to store input X's for each neuron
                    net = np.empty((self.inputShape[2], self.sizeOfKernels, self.sizeOfKernels))

                    # Then iterate over each relevant element that current kernel is observing
                    # (This locates the specific element from input in the area of each kernel)
                    for k in range(self.inputShape[2]):
                        for y in range(self.sizeOfKernels):
                            for x in range(self.sizeOfKernels):
                                net[k][y][x] = input[k][j + y][i + x]

                    # Insert results to the output matrix
                    out[z][j][i] = self.neurons[z][j][i].calculate(net)

        return out

    def calcwdeltas(self, wtimesdelta):
        # Create output matrix
        out = np.zeros((self.inputShape[2], self.inputShape[0], self.inputShape[1]))
        weight_sum = np.zeros((self.inputShape[2], self.sizeOfKernels, self.sizeOfKernels))
        bias_sum = np.zeros((self.numberOfKernels))

        for k in range(self.numberOfKernels):
            for i in range(self.outputX):
                for j in range(self.outputY):

                    d_out = self.neurons[k][i][j].calcpartialderivative(wtimesdelta[k][i][j], mode='Convolution')
                    w_up, b_up = self.neurons[k][i][j].updateweight(mode='Convolution')
                    np.add(weight_sum, w_up)
                    bias_sum[k] += b_up
                    for ch in range(self.inputShape[2]):
                        for row in range(self.sizeOfKernels):
                            for col in range(self.sizeOfKernels):
                                out[ch][i+row][j+col] += d_out[ch][row][col]

        for k in range(self.numberOfKernels):
            for i in range(self.outputX):
                for j in range(self.outputY):
                    self.neurons[k][i][j].bias -= bias_sum[k]
                    for ch in range(self.inputShape[2]):
                        for row in range(self.sizeOfKernels):
                            for col in range(self.sizeOfKernels):
                                self.neurons[k][i][j].weights[ch][row][col] -= weight_sum[ch][row][col]

        for kern in range(self.numberOfKernels):
            print(f'\nconvolutional layer, kernel {kern} weights:')
            print(self.neurons[kern][0][0].weights)

            print(f'\nconvolutional layer, kernel {kern} bias:')
            print(self.neurons[kern][0][0].bias)

        return out


class FlattenLayer:
    def __init__(self, inputSize):
        self.inputSize = inputSize
        self.numberOfNeurons = self.inputSize[0] * self.inputSize[1] * self.inputSize[2]
        self.outputShape = (self.numberOfNeurons)

    def calculate(self, input):
        flat = np.copy(input)

        flat = np.ravel(flat) 
        self.outputShape = flat.shape

        return np.array([flat])

    def calcwdeltas(self, wtimesdelta):
        # print(f'FLATTEN INPUT SIZE: {self.inputSize}')
        out_wd = np.empty((self.inputSize[2], self.inputSize[0], self.inputSize[1]))

        for k in range(self.inputSize[2]):
            for j in range(self.inputSize[0]):
                for i in range(self.inputSize[1]):
                    wtd_index = (k*self.inputSize[1]*self.inputSize[0]) + j*self.inputSize[0] + i
                    out_wd[k][j][i] = wtimesdelta[wtd_index]

        return out_wd


# An entire neural network
class NeuralNetwork:
    # initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    # def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
    def __init__(self, inputSize, loss, lr):
        # self.numOfLayers = None
        # self.numOfNeurons = numOfNeurons
        self.inputSize = inputSize
        # self.activation = activation
        self.loss = loss
        self.lr = lr
        # self.weights = weights

        self.input = None
        self.output = None
        self.layers = []

    def addLayer(self, layerType, inputSize=None, numberOfNeurons=None, numberOfKernels=None, sizeOfKernels=None,
                 activation=None, inputShape=None, weights=None, biases=None):

        print('Adding layer...')
        if len(self.layers) == 0:
            inSize = self.inputSize
        else:
            # I think this is what we are supposed to be getting? "Input size should be set to the current final layer"?
            inSize = self.layers[len(self.layers)-1].numberOfNeurons

        if inputShape is None and len(self.layers) == 0:
            inputShape = (inSize, inSize, 1)
        else:
            inputShape = self.layers[len(self.layers)-1].outputShape

        # def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        if layerType == "FullyConnected" or layerType == "fullyconnected":
            print(f'Fully Connected Input Size: {inSize}')
            if weights is None:
                self.layers.append(FullyConnected(numberOfNeurons, activation, inSize, self.lr))
            else:
                self.layers.append(FullyConnected(numberOfNeurons, activation, inSize, self.lr, weights, biases))

            print('Fully connected layer added.')

        elif layerType == "Conv" or layerType == "conv":
            print('Convolutional layer added.')
            if weights is None:
                self.layers.append(ConvolutionalLayer(numberOfKernels, sizeOfKernels, activation, inputShape, self.lr))
            else:
                self.layers.append(ConvolutionalLayer(numberOfKernels, sizeOfKernels, activation, inputShape,
                                                      self.lr, weights, biases))

        elif layerType == "MaxPool" or layerType == "maxpool":
            self.layers.append(MaxPoolingLayer(sizeOfKernels, inputShape))

            print('Max pooling layer added.')

        elif layerType == "Flatten" or layerType == "flatten":
            print('Flatten layer added. InShape = {0}'.format(inputShape))
            self.layers.append(FlattenLayer(inputShape))

        else:
            print('Layer could not be added.')

    # Given an input, calculate the output (using the layers calculate() method)
    def calculate(self, input):
        self.input = input   # Store list of inputs
        nextInput = self.input

        for layer in self.layers:
            nextInput = layer.calculate(nextInput)  # Store output of each layer as input into next layer

        self.output = nextInput
        return self.output    # Return last layer's output
        
    # Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, yp, y):
        if self.loss == 0:
            
            sum = 0     # Keep running sum
            for i in range(len(y)):
                sum += (y[i] - yp[i]) ** 2  # Square and add to total

            return sum / len(y)              

        elif self.loss == 1:
            # Do binary cross entropy
            sum = 0 
            for i in range(len(y)):
                sum += -1 * (y[i] * np.log(yp[i]) + ((1 - y[i]) * np.log(1-yp[i])))    # np.log is natural log 

            return sum / len(y)     # Online this showed to be an average


    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)
    def lossderiv(self, yp, y):
        numOfNeurons = len(self.layers[-1].neurons)     # Number of neurons in last layer
        pd_loss = []
        if self.loss == 0:
            for i in range(numOfNeurons):
                pd_loss.append(-2 * (y[i] - yp[i]))        # Current loss derivative set to use

        elif self.loss == 1:
            # Do binary cross entropy deriv 
            for i in range(numOfNeurons):
                pd_loss.append(((1 - y[i])/(1 - yp[i])) - (y[i] / yp[i]))   

        return pd_loss
    
    # Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values
    def train(self, x, y):
        # Number of layers in output layer
        numOfLayers = len(self.layers)

        y_test = self.calculate(x)      # One forward pass

        print(f'\nmodel output before: {y_test}')
        wtimesdelta = self.lossderiv(y_test, y)  # Save partial derivative of the loss as first w times delta
        print(f'Calc loss: {self.calculateloss(y_test,y)}')

        print(f'\n\nWeights after backpropagation.')
        for i in range(numOfLayers):
            curr_layer = numOfLayers - 1 - i       # Calc index for moving backwards
            wtimesdelta = self.layers[curr_layer].calcwdeltas(wtimesdelta)

        new_y_test = self.calculate(x)
        calc_loss = self.calculateloss(new_y_test, y)

        return new_y_test, calc_loss


def plot_lr(lr_list, labels, l=0, a=0):
    for lr in lr_list:
        plt.plot(range(len(lr)), lr)

    plt.legend(labels, loc='upper right')
    plt.title('Loss over {} Epochs; {} activation; {} loss'.format(EPOCHS, activations[a], losses[l]))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    x = np.array([[[0.1650159, 0.39252924, 0.09346037, 0.82110566, 0.15115202, 0.38411445, 0.94426071],
                   [0.98762547, 0.45630455, 0.82612284, 0.25137413, 0.59737165, 0.90283176, 0.53455795],
                   [0.59020136, 0.03928177, 0.35718176, 0.07961309, 0.30545992, 0.33071931, 0.7738303],
                   [0.03995921, 0.42949218, 0.31492687, 0.63649114, 0.34634715, 0.04309736, 0.87991517],
                   [0.76324059, 0.87809664, 0.41750914, 0.60557756, 0.51346663, 0.59783665, 0.26221566],
                   [0.30087131, 0.02539978, 0.30306256, 0.24207588, 0.55757819, 0.56550702, 0.47513225],
                   [0.29279798, 0.06425106, 0.97881915, 0.33970784, 0.49504863, 0.97708073, 0.44077382]]])

    y = np.array([0.31827281])

    # Weights for first conv layer, first kernel
    conv1_k1_weights = np.array([[[0.77132, 0.02075, 0.63365],
                                  [0.7488, 0.49851, 0.2248],
                                  [0.19806, 0.76053, 0.16911]]])

    conv1_k1_bias = 0.91777414

    # Weights for first conv layer, second kernel
    conv1_k2_weights = np.array([[[0.08834, 0.68536, 0.95339],
                                  [0.00395, 0.51219, 0.81262],
                                  [0.61253, 0.72176, 0.29188]]])

    conv1_k2_bias = 0.71457577

    conv1_biases = np.array([conv1_k1_bias, conv1_k2_bias])

    # Need to not be lists so the more we can keep in numpy arrays the better
    conv1_weights = np.array([conv1_k1_weights, conv1_k2_weights])

    # Weights for second conv layer w/ two channel
    conv2_k1_weights = [[[[0.54254, 0.14217, 0.37334],
                          [0.67413, 0.44183, 0.43401],
                          [0.61777, 0.51314, 0.6504]],
                         [[0.60104, 0.80522, 0.52165],
                          [0.90865, 0.31924, 0.09046],
                          [0.3007, 0.11398, 0.82868]]]]

    conv2_k1_bias = 0.04689632
    conv2_biases = np.array([conv2_k1_bias])

    conv2_weights = np.array(conv2_k1_weights)

    # Weights for flatten/dense layer
    fc_weights = np.array([[[0.62629, 0.54759, 0.81929, 0.19895, 0.85685, 0.35165, 0.75465, 0.29596, 0.88394]]])
    fc_bias = 0.32551163

    if len(sys.argv) < 2:
        print('Invalid input')

    elif sys.argv[1] == 'example1':
        print('Run example1.\n')

        NN = NeuralNetwork(5, MSE, 100)
        print('Initialized')
        NN.addLayer(layerType="Conv", numberOfKernels=1, sizeOfKernels=3, activation=SIGMOID, weights=conv1_weights,
                    biases=conv1_biases)
        NN.addLayer(layerType="Flatten")
        NN.addLayer(layerType="FullyConnected", numberOfNeurons=1, activation=SIGMOID, weights=fc_weights,
                    biases=fc_bias)

        output, loss = NN.train(x, y)
        print(f'\nOUTPUT: {output}')

    elif sys.argv[1] == 'example2':
        print('Run example2.\n')

        NN = NeuralNetwork(7, MSE, 100)
        print('Initialized')
        NN.addLayer(layerType="Conv", numberOfKernels=2, sizeOfKernels=3, activation=SIGMOID, weights=conv1_weights, biases=conv1_biases)
        NN.addLayer(layerType="Conv", numberOfKernels=1, sizeOfKernels=3, activation=SIGMOID, weights=conv2_weights, biases=conv2_biases)
        NN.addLayer(layerType="Flatten")
        NN.addLayer(layerType="FullyConnected", numberOfNeurons=1, activation=SIGMOID, weights=fc_weights, biases = fc_bias)

        output, loss = NN.train(x, y)
        print(f'\nOUTPUT: {output}')

    elif sys.argv[1] == 'example3':
        print('Run example3.\n')

        x = np.array([[[0.82612284, 0.25137413, 0.59737165, 0.90283176, 0.53455795,  0.59020136, 0.03928177, 0.35718176],
                       [0.07961309, 0.30545992, 0.33071931, 0.7738303, 0.03995921, 0.42949218, 0.31492687, 0.63649114],
                       [0.34634715, 0.04309736, 0.87991517, 0.76324059, 0.87809664, 0.41750914, 0.60557756, 0.51346663],
                       [0.59783665, 0.26221566, 0.30087131, 0.02539978, 0.30306256, 0.24207588, 0.55757819, 0.56550702],
                       [0.47513225, 0.29279798, 0.06425106, 0.97881915, 0.33970784, 0.49504863, 0.97708073, 0.44077382],
                       [0.31827281, 0.51979699, 0.57813643, 0.85393375, 0.06809727, 0.46453081, 0.78194912, 0.71860281],
                       [0.58602198, 0.03709441, 0.35065639, 0.56319068, 0.29972987, 0.51233415, 0.67346693, 0.15919373],
                       [0.05047767, 0.33781589, 0.10806377, 0.17890281, 0.8858271, 0.36536497, 0.21876935, 0.75249617]]])

        y = np.array([0.10687958])

        fc_weights = np.array([[[0.62628715, 0.54758616, 0.819287, 0.19894754, 0.8568503, 0.35165264,
                                 0.75464769, 0.29596171, 0.88393648, 0.32551164, 0.1650159, 0.39252924,
                                 0.09346037, 0.82110566, 0.15115202, 0.38411445, 0.94426071, 0.98762547]]])

        fc_bias = 0.45630455

        NN = NeuralNetwork(8, MSE, 100)
        print('Initialized')
        NN.addLayer(layerType="Conv", numberOfKernels=2, sizeOfKernels=3, activation=SIGMOID, weights=conv1_weights,
                    biases=conv1_biases)
        NN.addLayer(layerType="MaxPool", sizeOfKernels=2)
        NN.addLayer(layerType="Flatten")
        NN.addLayer(layerType="FullyConnected", numberOfNeurons=1, activation=SIGMOID, weights=fc_weights,
                    biases=fc_bias)

        output, loss = NN.train(x, y)
        print(f'\nOUTPUT: {output}')
