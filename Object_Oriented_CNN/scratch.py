import numpy as np
import sys
import random  # Only necessary if random initialized weights
import matplotlib.pyplot as plt


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
    # initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self, activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr

        self.d = 0
        self.input = None
        self.output = None

        # At the individual Neuron level if no weights specified
        # initialize the weights to random floating point values
        # in range(0.0 - 1.0)
        if weights is None:
            self.weights = [random.random() for i in range(input_num + 1)]
        else:
            self.weights = weights  # is a vector

    # This method returns the activation of the net
    def activate(self, net):
        if self.activation == 0:  # Linear Activation
            # ∂(x) = x
            return net
        elif self.activation == 1:  # Logistic Activation
            # ∂(x) = 1/(1+e^-x)
            exp = np.exp(-net)
            return float(1 / (1 + float(exp)))

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
        self.input.append(1)

        # print(self.input)
        # print(self.weights)
        mul = np.multiply(self.input, self.weights)
        net = np.sum(mul)
        # print()
        # print(self.weights)
        # print()
        # print(net)

        self.output = self.activate(net)

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        if self.activation == 0:
            # d(linear) / d(net) = constant
            return 1
        elif self.activation == 1:
            # d(logistic) / d(net) = logistic(net) * 1 - logistic(net)
            return self.output * (1 - self.output)

    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous
    # layer
    def calcpartialderivative(self, wtimesdelta):
        new_wd = []

        curr_delta = wtimesdelta * self.activationderivative()
        self.d = curr_delta

        for i in range(self.input_num + 1):
            new_wd.append(self.weights[i] * curr_delta)

        return new_wd

        # Simply update the weights using the partial derivatives and the learning weight

    def updateweight(self):
        for i in range(self.input_num + 1):
            self.weights[i] -= self.lr * self.d * self.input[i]


# A fully connected layer
class FullyConnected:
    # initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d
    # matrix of weights (or else initialize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
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
                self.neurons.append(Neuron(activation, input_num, lr, weights[i]))

    # calculate the output of all the neurons in the layer and return a vector with those values (go through the neurons
    # and call the calculate() method)
    def calculate(self, input):
        self.input = input
        self.output = []  # Reset array for output since it was infinitely expanding

        # Loop through each neuron and pass all Inputs
        for neuron in self.neurons:
            neuron.calculate(self.input)
            self.output.append(neuron.output)

        return self.output  # Send outputs back to neuralnetwork

    # given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the
    # correct value), sum up its ownw*delta, and then update the weights (using the updateweight() method). I should
    # return the sum of w*delta.
    def calcwdeltas(self, wtimesdelta):
        sum_wdelta = []

        for i in range(self.numOfNeurons):
            # Get each neuron's new wtimesdelta, not sure if the whole vector needs to be passed but it looks like it
            new_wd = self.neurons[i].calcpartialderivative(wtimesdelta[i])
            self.neurons[i].updateweight()

            if i == 0:
                sum_wdelta = new_wd  # If the vector doesn't exist yet, give it the first values
            else:
                sum_wdelta = np.add(sum_wdelta, new_wd)  # Otherwise, add vectors and keep a running sum

        return sum_wdelta


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
        self.kernels = []

        # Calculate size of output
        self.outputX = inputShape[0] - sizeOfKernels + 1
        self.outputY = inputShape[1] - sizeOfKernels + 1

        self.outputShape = (self.outputX, self.outputY, numberOfKernels)
        # self.numberOfNeurons = self.outputX * self.outputY * numberOfKernels
        self.numberOfNeurons = self.outputX * self.outputY

        for k in range(self.numberOfKernels):
            # Instantiate all Neurons in the given layer
            neurons = []
            for i in range(self.numberOfNeurons):
                if weights is None:
                    neurons.append(Neuron(activation, sizeOfKernels ** 2, lr, weights))  # self.numberOfNeurons
                else:                                                                           # was inputSize
                    # print('weights')
                    # print(weights[k])
                    neurons.append(Neuron(activation, sizeOfKernels ** 2, lr, weights[k]))

            self.kernels.append(neurons)

    def calculate(self, input):
        # Define output matrix
        out = np.empty((self.outputY, self.outputX, self.numberOfKernels), dtype=Neuron)

        # Loop for all channels
        # print(self.inputShape)
        ch_out = []
        for ch in range(self.inputShape[2]):
            k_out = []
            # Look at each kernel
            for k in self.kernels:
                out = []
                # for each node in the kernel
                for j in range(self.outputY):
                    net = []

                    # Then iterate over each relevant element that current kernel is observing
                    # (This locates the specific element from input in the area of each kernel)
                    for y in range(self.sizeOfKernels):
                        for x in range(self.sizeOfKernels):
                            net.append(input[ch][y][x])             ************************

                    k[j].calculate(net)
                    out.append(k[j].output)
                k_out.append(out)
            ch_out.append(k_out)
        return np.array(ch_out)

        # # Loop and for each neuron in all channels (This identifies where the kernel is looking)
        # for k in range(self.inputShape[2]):
        #     # j is row
        #     for j in range(self.outputY):
        #         # i is column
        #         for i in range(self.outputX):
        #             # Create a matrix to store input X's for each neuron
        #             # net = np.empty((self.sizeOfKernels, self.sizeOfKernels))
        #             net = []
        #
        #             # Then iterate over each relevant element that current kernel is observing
        #             # (This locates the specific element from input in the area of each kernel)
        #             for y in range(self.sizeOfKernels):
        #                 for x in range(self.sizeOfKernels):
        #                     # print(f'{k} {j} {i} {x} {y}')
        #                     # print(j * self.sizeOfKernels + y)
        #                     # net[y][x] = input[k][j * self.sizeOfKernels + y][i * self.sizeOfKernels + x]
        #                     net.append(input[k][j * self.sizeOfKernels + y][i * self.sizeOfKernels + x])
        #
        #             # Insert results to the output matrix
        #             print(net)
        #             print()
        #             print('conv neuron calculate')
        #             out[k][j][i] = self.neurons[k][j][i].calculate(net)
        # return out


class FlattenLayer:
    def __init__(self, inputSize):
        self.inputSize = inputSize
        self.numberOfNeurons = None
        self.outputShape = None

    def calculate(self, input):
        flat = np.copy(input)

        # If size is given as the shape, calculate size in one dimension
        # flat_size = inputSize[0] * inputSize[1] * inputSize[2]

        # If size is already 1D simply reshape
        np.reshape(flat, self.inputSize)
        self.numberOfNeurons = flat
        self.outputShape = flat

        return flat

    def calculatewdeltas(self, wtimesdelta):
        return


# An entire neural network
class NeuralNetwork:
    # initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each
    # layer), the loss function, the learning rate and a 3d matrix of weights (or else initialize randomly)
    # def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
    def __init__(self, inputSize, loss, lr):
        # self.numOfLayers = numOfLayers
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
                 activation=None, inputShape=None, weights=None):
        # (numberOfKernels, sizeOfKernels, activation, inputShape, lr, weights=None)
        # Input size should be set to the current final layer
        # I think it is supposed to be like model.add()?
        # model.add(layers.Conv2D(2,3,input_shape=(7,7,1),activation='sigmoid'))
        print('Adding layer...')
        if len(self.layers) == 0:
            inSize = self.inputSize
        else:
            # I think this is what we are supposed to be getting? "Input size should be set to the current final layer"?
            inSize = self.layers[len(self.layers) - 1].numberOfNeurons

        if inputShape is None and len(self.layers) == 0:
            inputShape = (inSize, inSize, 1)
        else:
            inputShape = self.layers[len(self.layers) - 1].outputShape

        if layerType == "FullyConnected" or layerType == "fullyconnected":
            if weights is None:
                self.layers.append(FullyConnected(numberOfNeurons, activation, inSize, self.lr))
            else:
                self.layers.append(FullyConnected(numberOfNeurons, activation, inSize, self.lr, weights))

            print('Fully connected layer added.')

        elif layerType == "Conv" or layerType == "conv":
            print('Convolutional layer added.')
            if weights is None:
                self.layers.append(ConvolutionalLayer(numberOfKernels, sizeOfKernels, activation, inputShape, self.lr))
            else:
                self.layers.append(ConvolutionalLayer(numberOfKernels, sizeOfKernels, activation, inputShape,
                                                      self.lr, weights))

        elif layerType == "MaxPool" or layerType == "maxpool":
            self.layers.append(MaxPoolingLayer(sizeOfKernels, inputShape))

            print('Max pooling layer added.')

        elif layerType == "Flatten" or layerType == "flatten":
            print('Flatten layer added.')
            self.layers.append(FlattenLayer(inSize))

        else:
            print('Layer could not be added.')

    # Given an input, calculate the output (using the layers calculate() method)
    def calculate(self, input):
        self.input = input  # Store list of inputs
        nextInput = self.input

        for layer in self.layers:
            print('Calculate next layer')
            nextInput = layer.calculate(nextInput)  # Store output of each layer as input into next layer
            print(nextInput)
            # exit()

        self.output = nextInput
        return self.output  # Return last layer's output

    # Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, yp, y):
        if self.loss == 0:

            sum = 0  # Keep running sum
            for i in range(len(y)):
                sum += (y[i] - yp[i]) ** 2  # Square and add to total

            return sum / len(y)

        elif self.loss == 1:
            # Do binary cross entropy
            sum = 0
            for i in range(len(y)):
                sum += -1 * (y[i] * np.log(yp[i]) + ((1 - y[i]) * np.log(1 - yp[i])))  # np.log is natural log

            return sum / len(y)  # Online this showed to be an average

    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the
    # loss function)
    def lossderiv(self, yp, y):

        pd_loss = []
        if self.loss == 0:
            for i in range(self.numOfNeurons[-1]):
                pd_loss.append(-1 * (y[i] - yp[i]))  # Current loss derivative set to use

        elif self.loss == 1:
            # Do binary cross entropy deriv
            for i in range(self.numOfNeurons[-1]):
                pd_loss.append(((1 - y[i]) / (1 - yp[i])) - (y[i] / yp[i]))

        return pd_loss

    # Given a single input and desired output preform one step of backpropagation (including a forward pass, getting
    # the derivative of the loss, and then calling calcwdeltas for layers with the right values
    def train(self, x, y):

        y_test = self.calculate(x)  # One forward pass

        wtimesdelta = self.lossderiv(y_test, y)  # Save partial derivative of the loss as first w times delta

        for i in range(self.numOfLayers):
            curr_layer = self.numOfLayers - 1 - i  # Calc index for moving backwards
            wtimesdelta = self.layers[curr_layer].calcwdeltas(wtimesdelta)

        new_y_test = self.calculate(x)
        calc_loss = self.calculateloss(new_y_test, y)

        return y_test, calc_loss


def plot_lr(lr_list, labels, l=0, a=0):
    for lr in lr_list:
        plt.plot(range(len(lr)), lr)

    plt.legend(labels, loc='upper right')
    plt.title('Loss over {} Epochs; {} activation; {} loss'.format(EPOCHS, activations[a], losses[l]))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Invalid input')

    elif sys.argv[1] == 'example1':
        print('Run example1.')

        x = np.array([[[0.1650159, 0.39252924, 0.09346037, 0.82110566, 0.15115202, 0.38411445, 0.94426071],
                     [0.98762547, 0.45630455, 0.82612284, 0.25137413, 0.59737165, 0.90283176, 0.53455795],
                     [0.59020136, 0.03928177, 0.35718176, 0.07961309, 0.30545992, 0.33071931, 0.7738303],
                     [0.03995921, 0.42949218, 0.31492687, 0.63649114, 0.34634715, 0.04309736, 0.87991517],
                     [0.76324059, 0.87809664, 0.41750914, 0.60557756, 0.51346663, 0.59783665, 0.26221566],
                     [0.30087131, 0.02539978, 0.30306256, 0.24207588, 0.55757819, 0.56550702, 0.47513225],
                     [0.29279798, 0.06425106, 0.97881915, 0.33970784, 0.49504863, 0.97708073, 0.44077382]]])

        y = np.array([0.31827281])

        # Changed arrays to be proper orientation should allow viewing better
        conv1_k1_neuron = [0.77126, 0.02068, 0.63358, 0.74873, 0.49844, 0.22472, 0.19798, 0.76046, 0.16903, 0.9176043]

        conv1_k2_neuron = [0.08828, 0.6853, 0.95333, 0.00388, 0.51212, 0.81255, 0.61246, 0.72169, 0.2918, 0.71441317]

        # Need to not be lists so the more we can keep in numpy arrays the better
        conv1_weights = np.array([[conv1_k1_neuron], [conv1_k2_neuron]])

        conv2_k1_neuron = [[0.54199, 0.14161, 0.37278, 0.67358, 0.44127, 0.43345, 0.61721, 0.51258, 0.64983, 0.04629412],
                           [0.60048, 0.80466, 0.52108, 0.90809, 0.31867, 0.08989, 0.30014, 0.11342, 0.82811, 0.04629412]]

        conv2_weights = np.array(conv2_k1_neuron)

        fc_weights = [0.15698, 0.07829, 0.34998, -0.27036, 0.38755, -0.11766, 0.28534, -0.17335, 0.41462]
        fc_bias = -0.14390945

        fc_weights.append(fc_bias)

        # run a network with a 5x5 input, one 3x3 convolution layer
        # with a single kernel, a flatten layer, and single neuron for the output
        # example2 uses sigmoid, MSE, and learning rate 100
        NN = NeuralNetwork(7, MSE, 100)
        print('Initialized')
        NN.addLayer(layerType="Conv", numberOfKernels=2, sizeOfKernels=3, activation=SIGMOID, weights=conv1_weights)
        NN.addLayer(layerType="Conv", numberOfKernels=1, sizeOfKernels=3, activation=SIGMOID, weights=conv2_weights)
        NN.addLayer(layerType="Flatten")
        NN.addLayer(layerType="FullyConnected", numberOfNeurons=1, activation=SIGMOID, weights=fc_weights)

        output, loss = NN.train(x, y)
