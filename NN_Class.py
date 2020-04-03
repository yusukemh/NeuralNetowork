import numpy as np

class NeuralNetwork:
    
    def __init__(self, hidden_layer_sizes = (3,), activation='relu'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.n_layers = self.n_hidden_layers + 2
        self.activation = activation
        #self.initializeWeights()
    
    def initializeWeights(self, X):
        #layers = [801, 100, 1]
        self.W1 = np.random.rand(layers[0], layers[1]) / 100
        self.W2 = np.random.rand(layers[1], layers[2]) / 100
        
        '''
        self.w = []
        
        
        #the first layer (btwn the input and the first hidden layer)
        self.w.append(np.arange(len(X) * self.hidden_layer_sizes[0]).reshape((len(X), -1)))
        
        #weights between hidden layers
        for i in range(1, self.n_hidden_layers):
            self.w.append(np.arange(self.hidden_layer_sizes[i - 1] * self.hidden_layer_sizes[i]).reshape((self.hidden_layer_sizes[i - 1], -1)))
            pass
        
        #the last layer (btwn the last hidden layer and the output)
        self.w.append(np.arange(self.hidden_layer_sizes[-1]).reshape((self.hidden_layer_sizes[-1], -1)))
        
        #self.w = np.array(self.w)'''
        pass
    
    def fit(self, X, y, iter=100):
        #initialize the weights
        self.initializeWeights(X)
        
        #forward
        
        pass
    
    #prof's code
    def predict_proba(self, x):
        z1 = np.dot(x, self.W1)
        
        z1 = np.dor(np.hstack(x, np.ones((x.shape[0], 1))), self.W1)
        
        if self.activation == 'relu':
            h1 = np.maximum(0, z1)
        else:
            raise IOError('Please use valid activation function.')
            
        z2 = np.dot(h1, self.W2)
        y_pred = 1. / (1+np.exp(-z2))
        return y_pred
        pass
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    
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
                
    
    
    '''
    W = {}
    for i in range(len(layers) - 1):
        W[i] = np.random.rand(layers[i], layers[i+1]) / 100
    self.W = W
    ===
    dL_dz1 = # N x 100
    dW1 = np.dot(X.T, dL_dz1) # 801 x N dot N x 100 = 801 x 100 //sum N of products
    '''