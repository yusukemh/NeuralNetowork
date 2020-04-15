import numpy as np
#import NN_Class
import NeuralNetwork
N = 100
D = 5
X = np.random.rand(N, D) # NxD matrix
Y = np.zeros(N) # length N vector

model = NeuralNetwork.NNClassifier([6,10])
model.fit(X, Y)
a = np.array([1,2,3])
print(np.multiply(4,a))



#model.printWeights()
#print(len(model.getHiddenLayerSizes()))
#print(model.getNumLayers())