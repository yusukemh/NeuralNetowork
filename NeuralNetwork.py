import numpy as np

class NNClassifier():
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.n_layers = self.n_hidden_layers + 2
        self.W = {}
        self.layer_sizes = []
        #Randomly initialize the weights.
        #Since the input dimension is unknown until x is passed at fit(), only initialize
        #   weights from the second layer to the output layer.
        '''
        ===
        dL_dz1 = # N x 100
        dW1 = np.dot(X.T, dL_dz1) # 801 x N dot N x 100 = 801 x 100 //sum N of products
        '''
        for i in range(1, self.n_hidden_layers):#leave room for the weight for the first layer
            self.W[i] = np.random.rand(hidden_layer_sizes[i-1] + 1, hidden_layer_sizes[i]) / 100 
            # -> (layers[i]+1) x layers[i+1] matrix: bias term included
            
        #add the weight matrix for the last layer: bias term included
        self.W[self.n_hidden_layers] = np.random.rand(hidden_layer_sizes[-1] + 1, 1) / 100
        #-> (layers[i]+1) x 1 matrix
        
            
    def fit(self, X, Y):
        #train using input data
        #Input: features x: NxD matrix
        #Output: labels y, a length N vector of 0s, 1s
        
        #first, add the weight matrix for the layer btwn the input and the first layer
        self.W[0] = np.random.rand(X.shape[1] + 1,  self.hidden_layer_sizes[0]) / 100
        # -> (D + 1) x hidden_layer_sizes[0] matrix: bias term included
        
        n = X.shape[0]
        bias = np.ones((n,1))
        
        for i in range(self.n_hidden_layers + 1):
            print(self.W[i].shape)
            print(self.W[i])
            print('\n')
        
        #forward propagation
        Z = {}
        H = {}
        #consider the input layer as Z[0] for simplisity.
        Z[0] = X
        #do not apply activation for the input layer, but add a column of one's
        H[0] = np.hstack((Z[0], bias))
        
        #propagate through each layer
        for i in range(1, self.n_hidden_layers + 1): #1,2,..., n_hidden_layers
            #calculate the sum
            Z[i] = np.dot(H[i-1], self.W[i-1])
            #add column
            H[i] = np.hstack((self.relu(Z[i]), bias))
        Yhat = np.dot(H[self.n_hidden_layers], self.W[self.n_hidden_layers])
        Yhat = self.sigmoid(Yhat)
        print(Yhat)
        pass
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def relu(self, x):#checked
        return np.maximum(np.zeros((x.shape[0],x.shape[1])), x)
    
    def predict_proba(self, x):
        pass