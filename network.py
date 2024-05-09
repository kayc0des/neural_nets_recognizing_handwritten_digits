''' A network class: used to represent a neural network'''

import numpy as np

# network class
class Network(object):
    
    def __init__(self, sizes):
        '''
        Constructor method
        
        Param: sizes - List contains the number of neurons in the respective layers.
        '''
        self.num_layers = len(sizes) #the number of layers = to the length of the list
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # biases are ignored for the input layer
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
        
# debug
net = Network([2, 3, 1])
print(net.weights)
print(net.biases)