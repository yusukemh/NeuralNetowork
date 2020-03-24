import numpy as np
import NN_Class

a = np.array([2,3])

model = NN_Class.NeuralNetwork()
model.fit(a, a)

model.printWeights()
#print(len(model.getHiddenLayerSizes()))
#print(model.getNumLayers())