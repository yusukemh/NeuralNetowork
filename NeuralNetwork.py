import numpy as np
import math
np.set_printoptions(precision=10)
class NNClassifier():
    def __init__(self, hidden_layer_sizes, learning_rate = 0.08, batch_size = 200, epochs = 100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.n_layers = self.n_hidden_layers + 2
        self.batch_size = batch_size
        self.epochs = epochs
        self.W = {}
        self.Z = {}
        self.H = {}
        self.layer_sizes = []
        
            
    def fit(self, X, Y):
        _lambda = 0.01
        #train using input data
        #Input: features x: NxD matrix
        #Output: labels y, a length N vector of 0s, 1s
        
        #preprocess
        X_intact = X
        Y_intact = Y.reshape(-1,1)

        #incorporate the input and output layers
        self.layer_sizes.append(X.shape[1])
        for size in self.hidden_layer_sizes:
          self.layer_sizes.append(size)
        self.layer_sizes.append(1)
        #initialize the weight randomly. Allow negative values
        for i in range(self.n_layers - 1):
          self.W[i] = (np.random.rand(self.layer_sizes[i] + 1, self.layer_sizes[i+1]) - 0.5) / 100
          
        iteration = int(X.shape[0] / self.batch_size)
        n = self.batch_size
        column = np.ones((n,1))
        
        for epoch in range(self.epochs):
          start = 0
          end = self.batch_size
          
          for step in range(iteration):
            X_batch = X_intact[start:end, :]
            Y_batch = Y_intact[start:end, :]
            
            #FORWARD PROPAGATION
            self.Z[0] = X_batch   #treat the input layer as Z[0]
            self.H[0] = np.hstack((self.Z[0], column)) #add extra culumn
            for i in range(1, self.n_layers - 1):#Calculate hidden layers
              self.Z[i] = np.dot(self.H[i-1], self.W[i-1])
              self.H[i] = np.hstack((self.relu(self.Z[i]), column))
            Yhat = np.dot(self.H[self.n_layers - 2], self.W[self.n_layers - 2])
            Yhat = self.sigmoid(Yhat)
            
            #BACK PROPAGATION
            dLdz = Yhat - Y_batch# (N, 1)
            dLdw = np.dot(np.transpose(dLdz), self.H[self.n_layers - 2]) #(1, -1)
            #update the layer
            self.W[self.n_layers - 2] -= np.multiply(self.learning_rate, np.transpose(dLdw)) + np.multiply((_lambda), self.W[self.n_layers - 2])
            #G = {}
            #G[self.n_layers - 2] = np.transpose(dLdw)      
            for i in range(self.n_layers - 3, -1, -1): #n_layers - 3, n_layer - 4, ... 0
              dLdz = np.dot(dLdz, np.transpose(self.W[i+1][:-1,:]))
              dLdz = np.multiply(dLdz, self.reluD(self.Z[i+1]))
              dLdw = np.dot(np.transpose(dLdz), self.H[i])
              #G[i] = np.transpose(dLdw)
              self.W[i] -= np.multiply(self.learning_rate, np.transpose(dLdw)) + np.multiply((_lambda), self.W[i])
              #self.W[i] = np.clip(self.W[i], 1e-100, 1e100)
              #print(self.W[i][:10,:10])
              #if(np.isnan(np.sum(self.W[i]))):
                #print("NAN")
                 #return
              
            ''' 
            #Use the gradient to update the weights
            for i in range(self.n_layers -1):
              self.W[i] = np.subtract(self.W[i], np.multiply(self.learning_rate, G[i]))
            '''
            start += self.batch_size
            end += self.batch_size
        
            
        
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
    def relu(self, x):#checked
        return np.maximum(np.zeros((x.shape[0],x.shape[1])), x)
    
    def reluD(self, x):
        return np.maximum(np.zeros((x.shape[0], x.shape[1])), 1)
    
    def predict_proba(self, X):
        #FORWARD PROPAGATION
        n = X.shape[0]
        column = np.ones((n,1))
        self.Z[0] = X   #treat the input layer as Z[0]
        self.H[0] = np.hstack((self.Z[0], column)) #add extra culumn
        
        for i in range(1, self.n_layers - 1):#Calculate hidden layers
            self.Z[i] = np.dot(self.H[i-1], self.W[i-1])
            self.H[i] = np.hstack((self.relu(self.Z[i]), column))
        Yhat = np.dot(self.H[self.n_layers - 2], self.W[self.n_layers - 2])
        Yhat = self.sigmoid(Yhat)
        return Yhat