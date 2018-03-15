import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import seaborn as sns
import pandas as pd


from sklearn.ensemble import GradientBoostingClassifier

# import X training points with 15 features and Y training points
X_data = np.genfromtxt("X_train.txt",delimiter=None)
Y_data = np.genfromtxt("Y_train.txt",delimiter=None)

# import X testing points
X_test = np.genfromtxt("X_test.txt",delimiter=None)

X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)
Test = X_test.shape[0]


predict = np.zeros((Test, 2))




clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=1, max_depth=500
                              ,max_leaf_nodes=50)

clf.fit(X_train, Y_train)

Yhat = (clf.predict(X_valid))
Ythat = (clf.predict(X_train))
predict = clf.predict_proba(X_test)

print(predict)
print('Training Accuracy', np.mean(Ythat == Y_train))
print('Validation Accuracy', np.mean(Yhat == Y_valid))
np.savetxt('GB_predict.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T, 
            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

#combines a set of weak learners using a gradient descent-like procedure, outcomes are weighed based on the previous instant
#misclassifed outcomes will have higher weight

#Regression trees are used as weak learners, and their outputs are added together and correct the residuals in the predictions
#(based on a loss function of your choice). A gradient descent procedure is used to minimize the loss when
#adding the trees together.
#We put constraints on the trees to make sure they stay weak. We also weight the predictions of each tree to slow down the
#learning of the algorithm.

#learning rate: impact of each tree on the outcome. Magnitude of changes based
#on the outputs of the trees
#Lower values make the model robust to tree characteristics, and requires more
#trees to model all relations (makes it more expensive)

#n_estimators: number of sequential trees to be modeled
#GBM is pretty robust against overfitting, but still will at some point (should
#tune n_estimators for a particular learning rate)

#max_depth: limits the number of nodes in the tree
