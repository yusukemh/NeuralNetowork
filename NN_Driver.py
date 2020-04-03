import numpy as np
#import NN_Class
import NeuralNetwork
N = 3
D = 5
X = np.random.rand(N, D) # NxD matrix with N = 100, D = 18
Y = np.random.rand(N) # length N vector

model = NeuralNetwork.NNClassifier([10,10])
model.fit(X, Y)




#model.printWeights()
#print(len(model.getHiddenLayerSizes()))
#print(model.getNumLayers())