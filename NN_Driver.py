
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics # Area Under the ROC calculations.
import NeuralNetwork
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_tr = pd.read_csv('tox21_dense_train.csv')
x_tr = np.array(x_tr.iloc[:,1:])
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)

y_tr = pd.read_csv('tox21_labels_train.csv')
y_tr = y_tr.fillna(0)
y_tr = np.array(y_tr.loc[:,'NR.AR']) # Each Column is a type of toxicity.

x_te = pd.read_csv('tox21_dense_test.csv')
x_te = np.array(x_te.iloc[:,1:])
x_te = scaler.transform(x_te) # Must use exact same preprocessing as train set.

y_te = pd.read_csv('tox21_labels_test.csv')
y_te = y_te.fillna(0)
y_te = np.array(y_te.loc[:,'NR.AR']) 

'''
label = np.array(np.where(y_tr == 1)).flatten()
label_2 = np.array(np.where(y_tr == 0)).flatten()
a = np.concatenate((label, label_2[:383]))

#handle the unbalanced data
from sklearn.utils import resample
y_tr = y_tr.reshape(-1,1)
Train = np.hstack((y_tr, x_tr))
#print("train:", Train.shape)

y_te = y_te.reshape(-1,1)
Test = np.hstack((y_te, x_te))
#print("Test:", Test.shape)

x_tr_majority = Train[Train[:,0] == 0,:]
x_tr_minority = Train[Train[:,0] == 1,:]

#print(x_tr_majority.shape)
#print(x_tr_minority.shape)
minority_upsampled = resample(x_tr_minority, 
                              replace=True,     # sample with replacement
                              n_samples=11677,  # to match majority class
                              random_state=123) # reproducible results
Train = np.vstack((x_tr_majority, minority_upsampled))
#print(Train.shape)

y_tr_new = Train[:,0]
x_tr_new = Train[:,1:]
'''
model = NeuralNetwork.NNClassifier([30,30,30], learning_rate = 0.01, 
                                    batch_size = 100, epochs = 100, early_stopping = True,
                                    n_iter_no_change = 10, tol = 0.0001,
                                    random_state = 20)
#x_tr = x_tr[:,:]
#y_tr = y_tr[:]

x_tr = sklearn.preprocessing.normalize(x_tr)
x_te = sklearn.preprocessing.normalize(x_te)


#print(x_tr.shape)
model.fit(x_tr, y_tr)

p_tr = model.predict_proba(x_tr)
auc_tr = sklearn.metrics.roc_auc_score(y_tr, p_tr)
print("Train set AUC: %3.5f" % (auc_tr))
p_te = model.predict_proba(x_te)
auc_te = sklearn.metrics.roc_auc_score(y_te, p_te)
print("Test set AUC: %3.5f" % (auc_te))













'''
N = 10
D = 5
#X = np.random.rand(N, D) # NxD matrix
#Y = np.zeros(N) # length N vector

X = np.array([[1,1,0,1,1],
              [1,1,0,1,1],
              [1,1,0,0,1],
              [1,1,1,1,0],
              [1,0,1,1,1],
              [0,0,1,0,0],
              [0,0,1,1,0],
              [0,1,0,1,0],
              [0,0,1,0,1],
              [0,0,0,1,1]])
Y = np.array([[1],
              [1],
              [1],
              [1],
              [1],
              [0],
              [0],
              [0],
              [0],
              [0]])

model = NeuralNetwork.NNClassifier([50,50],learning_rate = 0.05, batch_size = 10, epochs = 1000)
model.fit(X, Y)
Yhat = model.predict_proba(X)
print("Afer all...")
print(Yhat)
'''




