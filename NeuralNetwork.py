import numpy as np
import math
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=10)
class NNClassifier():
    def __init__(self, hidden_layer_sizes, learning_rate = 0.08, batch_size = 200,
                epochs = 100, early_stopping = False,
                n_iter_no_change = 10, tol = 0.0001,
                random_state = -1):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.n_layers = self.n_hidden_layers + 2
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        if(random_state == -1):
            self.random_state = np.random.randomint()
        else:
            self.random_state = random_state
        self.tol = tol
        self.W = {}
        self.Z = {}
        self.H = {}
        self.layer_sizes = []
        
            
    def fit(self, X, Y):
        np.random.seed(self.random_state)
        _lambda = 0.01
        #train using input data
        #Input: features x: NxD matrix
        #Output: labels y, a length N vector of 0s, 1s
        
        #split the data
        if(self.early_stopping):
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1)
        else:
            X_train = X
            Y_train = Y
            X_valid = np.array([])
            Y_valid = np.array([])
        #preprocess
        Y_valid = Y_valid.reshape(-1,1)
        Y_train = Y_train.reshape(-1,1)
        
        #divide the dataset if early_stopping is set True

        #incorporate the input and output layers
        self.layer_sizes.append(X_train.shape[1])
        for size in self.hidden_layer_sizes:
          self.layer_sizes.append(size)
        self.layer_sizes.append(1)
        #initialize the weight randomly. Allow negative values
        for i in range(self.n_layers - 1):
          self.W[i] = (np.random.rand(self.layer_sizes[i] + 1, self.layer_sizes[i+1]) - 0.5) / 100
        
        
        iteration = int(X_train.shape[0] / self.batch_size)
        n = self.batch_size
        column = np.ones((n,1))
        loss_prev = 10000#variable for early stopping
        last_improve_epoch = 0#last epoch with improvement
        for epoch in range(self.epochs):
          #check for early stopping
          print("Epoch:", epoch+1, "/", self.epochs)
          if(self.early_stopping):
              loss_curr = self.cross_entropy(self.forward(X_valid), Y_valid)#calculate valid loss
              if(loss_prev - loss_curr < self.tol):#if no improvement observed
                  if(epoch - last_improve_epoch >= self.n_iter_no_change):#if reached the limit, return to caller
                      print("EARLY STOPPING")
                      return
              else:#if improvement observed
                  last_improve_epoch = epoch
              loss_prev = loss_curr
          
          start = 0
          end = self.batch_size
          
          for step in range(iteration):
            #set up the batch
            X_batch = X_train[start:end, :]
            Y_batch = Y_train[start:end, :]
            
            #FORWARD PROPAGATION
            Yhat = self.forward(X_batch)
            
            #BACK PROPAGATION
            dLdz = Yhat - Y_batch
            dLdw = np.dot(np.transpose(dLdz), self.H[self.n_layers - 2])
            #update the layer, use L2 regularization
            self.W[self.n_layers - 2] -= np.multiply(self.learning_rate, np.transpose(dLdw)) + np.multiply((_lambda), self.W[self.n_layers - 2])  
            for i in range(self.n_layers - 3, -1, -1): #n_layers - 3, n_layer - 4, ... 0
              dLdz = np.dot(dLdz, np.transpose(self.W[i+1][:-1,:]))
              dLdz = np.multiply(dLdz, self.reluD(self.Z[i+1]))
              dLdw = np.dot(np.transpose(dLdz), self.H[i])
              self.W[i] -= np.multiply(self.learning_rate, np.transpose(dLdw)) + np.multiply((_lambda), self.W[i])
            
            #update the batch
            start += self.batch_size
            end += self.batch_size
        
            
        
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
    def relu(self, x):#checked
        return np.maximum(np.zeros((x.shape[0],x.shape[1])), x)
    
    def reluD(self, x):
        return np.maximum(np.zeros((x.shape[0], x.shape[1])), 1)
    
    def forward(self, X):
        #FORWARD PROPAGATION
        #Returns: Yhat
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
        
    def cross_entropy(self, Yhat, Y):
        return -np.sum(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat)) / Yhat.shape[0]
    
    def predict_proba(self, X):
        return self.forward(X)