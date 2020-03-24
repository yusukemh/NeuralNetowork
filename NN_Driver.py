import numpy as np
import NN_Class

a = np.array([2,3,4])

model = NN_Class.NeuralNetwork(a)
print(model.getX())