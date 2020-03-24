import numpy as np

class NeuralNetwork:
    
    def __init__(self, hidden_layer_sizes = (3, 2)):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.n_layers = self.n_hidden_layers + 2
        #self.initializeWeights()
    
    def initializeWeights(self, X):
        self.w = []
        
        #the first layer (btwn the input and the first hidden layer)
        self.w.append(np.arange(len(X) * self.hidden_layer_sizes[0]).reshape((len(X), -1)))
        
        #weights between hidden layers
        for i in range(1, self.n_hidden_layers):
            self.w.append(np.arange(self.hidden_layer_sizes[i - 1] * self.hidden_layer_sizes[i]).reshape((self.hidden_layer_sizes[i - 1], -1)))
            pass
        
        #the last layer (btwn the last hidden layer and the output)
        self.w.append(np.arange(self.hidden_layer_sizes[-1]).reshape((self.hidden_layer_sizes[-1], -1)))
        
        #self.w = np.array(self.w)
        pass
    
    def fit(self, X, y):
        self.initializeWeights(X)
        pass
    
    #get functions
    def getHiddenLayerSizes(self):
        return self.hidden_layer_sizes
    
    def getNumLayers(self):
        return self.n_layers
    
    def printWeights(self):
        print(type(self.w))
        for weights in self.w:
            print(type(weights))
            for weight in weights:
                print(weight, "")