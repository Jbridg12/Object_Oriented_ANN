import numpy as np
import sys
import random   # Only necessary if random initialized weights
import matplotlib.pyplot as plt
import CNN

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

# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
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
        self.input.append(1)

        mul = np.multiply(self.input, self.weights)
        net = np.sum(mul)

        self.output = self.activate(net)

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        if self.activation == 0:
            # d(linear) / d(net) = constant
            return 1    
        elif self.activation == 1:
            # d(logistic) / d(net) = logistic(net) * 1 - logistic(net)
            return self.output * (1 - self.output)

    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        new_wd = []

        curr_delta = wtimesdelta * self.activationderivative()
        self.d = curr_delta

        for i in range(self.input_num + 1):
            new_wd.append( self.weights[i] * curr_delta)

        return new_wd 
    
    # Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        for i in range(self.input_num + 1):
            self.weights[i] -= self.lr * self.d * self.input[i]


# A fully connected layer
class FullyConnected:
    # initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
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

    # calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)
    def calculate(self, input):
        self.input = input
        self.output = []    # Reset array for ouput since it was infinitely expanding

        # Loop through each neuron and pass all Inputs 
        for neuron in self.neurons:
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

        return sum_wdelta           


# An entire neural network
class NeuralNetwork:
    # initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.inputSize = inputSize
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.weights = weights

        self.input = None
        self.output = None

        # I think this part needs to be taken out from what problem description 1a is saying???
        # List of numOfLayers Fully Connected elements
        # self.layers = []
        # for i in range(numOfLayers):
        #     inSize = None
        #     if i == 0:
        #         # if layer receives from input layer
        #         inSize = inputSize
        #     else:
        #         # if layer is not receiving from input layer, see how many neurons were in previous layer
        #         inSize = numOfNeurons[i-1]
        #
        #
        #     if weights is None:
        #         self.layers.append(FullyConnected(numOfNeurons[i], activation[i], inSize, lr))
        #     else:
        #         self.layers.append(FullyConnected(numOfNeurons[i], activation[i], inSize, lr, weights[i]))
    
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

        pd_loss = []
        if self.loss == 0:
            for i in range(self.numOfNeurons[-1]):
                pd_loss.append(-1 * (y[i] - yp[i]))        # Current loss derivative set to use

        elif self.loss == 1:
            # Do binary cross entropy deriv 
            for i in range(self.numOfNeurons[-1]):
                pd_loss.append(((1 - y[i])/(1 - yp[i])) - (y[i] / yp[i]))   

        return pd_loss
    
    # Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values
    def train(self, x, y):

        y_test = self.calculate(x)      # One forward pass
        
        wtimesdelta = self.lossderiv(y_test, y)  # Save partial derivative of the loss as first w times delta

        for i in range(self.numOfLayers):
            curr_layer = self.numOfLayers - 1 - i       # Calc index for moving backwards
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

    elif sys.argv[2] == 'example1':
        print('Run example from class (single step).')

        w = np.array([[[.15, .2, .35], [.25, .3, .35]], [[.4, .45, .6], [.5, .55, .6]]])
        x = np.array([0.05, 0.1])      
        y = np.array([0.01, 0.99])

        # Test neural network
        # numOfLayers(includes hidden and output layers),
        # numOfNeurons(an array with number for each layer),
        # inputSize,
        # activation(array with activation for each layer),
        # loss, lr, weights=None
        NN = NeuralNetwork(2, [2, 2], 2, [1, 1], 0, float(sys.argv[1]), w)

        for layer in range(NN.numOfLayers):
            print(f'{NN.layers[layer].weights}')

        output, loss = NN.train(x, y)

        print(f'\nOutput: {output}, Loss: {loss}')
        print()

        print('Weights after 1-step update.')
        for layer in range(NN.numOfLayers):
            print(f'{NN.layers[layer].weights}')

    elif sys.argv[2] == 'example2':
        exit()

    elif sys.argv[2] == 'example3':
        exit()
