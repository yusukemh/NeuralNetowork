
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics # Area Under the ROC calculations.
import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()

#import the data
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

#split train into train and valid
x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.01)
model = NeuralNetwork.NNClassifier([50,50], learning_rate = 0.01, 
                                    batch_size = 200, epochs = 100, early_stopping = True,
                                    n_iter_no_change = 10, tol = 0.001,
                                    random_state = 30, verbose = False)

x_tr = sklearn.preprocessing.normalize(x_tr)
x_val = sklearn.preprocessing.normalize(x_val)
x_te = sklearn.preprocessing.normalize(x_te)


#print(x_tr.shape)
model.fit(x_tr, y_tr)

p_tr = model.predict_proba(x_tr)
auc_tr = sklearn.metrics.roc_auc_score(y_tr, p_tr)
print("Train set AUC: %3.5f" % (auc_tr))

p_val = model.predict_proba(x_val)
auc_val = sklearn.metrics.roc_auc_score(y_val, p_val)
print("Valid set AUC: %3.5f" % (auc_val))

p_te = model.predict_proba(x_te)
auc_te = sklearn.metrics.roc_auc_score(y_te, p_te)
print("Test set AUC: %3.5f" % (auc_te))

