import numpy as np
import sys

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self, activation, input_num, lr, weights=None):
        print('constructor')
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights  # is a vector

        self.pd_weights = None # vector for backpropogation
        self.input = None   # is a vector bc weights is a vector?
        self.output = None  # is a vector bc weights is a vector? | I think this should be a single value right?

    # This method returns the activation of the net
    def activate(self, net):
        if self.activation == 0:    # Linear Activation
            # ∂(x) = x
            return net
        elif self.activation == 1:                   # Logistic Activation
            # ∂(x) = 1/(1+e^-x)
            exp = np.exp(-net)
            return float(1/(1+float(exp)))

        print('activate')   

    # Calculate the output of the neuron should save the input and output for back-propagation.
    def calculate(self, input):
        """
        1. multiply i*w
        2. add i*w + i*w
        3. apply activation function

        ? assuming we do not need to include a bias in calculation?
        :param input:
        :return:
        """
        self.input = input

        # Append 1 for bias
        input.append(1)

        mul = np.multiply(self.input, self.weights)
        net = np.sum(mul)

        self.output = self.activate(net)

        print('calculate')

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        if self.activation == 0:
            # d(linear) / d(net) = constant
            return 1    # I imagine this is just a constant right?
        elif self.activation == 1:
            # d(logistic) / d(net) = logistic(net) * 1 - logistic(net)
            return self.activate(self.output) * ( 1 - self.activate(self.output))

        print('activationderivative')   
    
    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):

        print('calcpartialderivative') 
    
    # Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        print('updateweight')

        
# A fully connected layer
class FullyConnected:
    # initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        print('constructor') 
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights

        # I imagine we need to instantiate the neurons somewhere but writeup doesn't specify so here?
        self.neurons = None
        # for i in range(numOfNeurons):
        #     self.neurons.Add(Neuron(activation, input_num, lr, weights))
        
    # calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)
    def calculate(self, input):

        out = None  # Store output of each neuron in list for feed forward

        # Loop through each neuron and pass all Inputs 
        for neuron in self.neurons:
            out.Add(neuron.calculate(input))
        
        print('calculate') 
        return out  # Send outputs back to neuralnetwork
        

    # given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.
    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas') 
           
        
# An entire neural network
class NeuralNetwork:
    # initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        print('constructor')
        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.inputSize = inputSize
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.weights = weights  # TODO: randomize weights if weights is None

        # I imagine we need to instantiate the layers somewhere but writeup doesn't specify so here?
        # List of numOfLayers Fully Connected elements
        self.layers = []
        for i in range(numOfLayers):
            inSize = None
            if i == 0:
                # if layer receives from input layer
                inSize = inputSize
            else:
                # if layer is not receiving from input layer, see how many neurons were in previous layer
                inSize = numOfNeurons[i-1]

            # self.layers.Add(FullyConnected(numOfNeurons, activation, inputSize, lr, weights))
            self.layers.append(FullyConnected(numOfNeurons[i], activation[i], inSize, lr, weights[i]))
    
    # Given an input, calculate the output (using the layers calculate() method)
    def calculate(self, input):
        
        nextInput = input   # Store list of inputs 

        # Definitely not quite how this should work but just implementing basics
        for layer in self.layers:
            nextInput = layer.calculate(nextInput) # Store output of each layer as input into next layer

        return nextInput    # Return last layer's output
        print('calculate')
        
    # Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, yp, y):
        if self.loss == 0:
            # Do sum of squares
            return
        elif self.loss == 1:
            # Do binary cross entropy
            return
        print('calculateloss')
    
    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)
    def lossderiv(self, yp, y):
        if self.loss == 0:
            return yp - y
        elif self.loss == 1:
            # Do binary cross entropy deriv 
            return (yp - y) / (yp * (1 - yp))
        print('lossderiv')
    
    # Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values
    def train(self, x, y):
        print('train')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # print('a good place to test different parts of your code')
        # Testing neuron -- Delete later
        # activation, input_num, lr, weights=None
        N = Neuron(1, 2, .01, [.15, .2, .35])
        N.calculate([.05, .1])
        print(N.output)

        # Test fully connected layer -- Delete later
        # numOfNeurons, activation, input_num, lr, weights=None
        layer = FullyConnected(2, 1, 2, .01, [.15, .2, .35])

    elif sys.argv[1] == 'example':
        print('run example from class (single step)')
        w = np.array([[[.15, .2, .35], [.25, .3, .35]], [[.4, .45, .6], [.5, .55, .6]]])
        x = np.array([0.05, 0.1])      # I think he meant =? So changed from x == np.array([0.05, 0.1])
        np.array([0.01, 0.99])

        # Test neural network
        # numOfLayers(includes hidden and output layers),
        # numOfNeurons(an array with number for each layer),
        # inputSize,
        # activation(array with activation for each layer),
        # loss, lr, weights=None
        NN = NeuralNetwork(2, [2, 2], 2, [1, 1], 0, 0.1, w)
        print(len(NN.layers))
        # "Train" network on input
        # NN.calculate(x)
        
    elif sys.argv[1] == 'and':
        print('learn and')
        
    elif sys.argv[1] == 'xor':
        print('learn xor')
