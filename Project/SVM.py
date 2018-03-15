import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import seaborn as sns
import pandas as pd

import sklearn.model_selection

# import X training points with 15 features and Y training points
X_data = np.genfromtxt("X_train.txt",delimiter=None)
Y_data = np.genfromtxt("Y_train.txt",delimiter=None)

# import X testing points
X_test = np.genfromtxt("X_test.txt",delimiter=None)

X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)

print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn import svm


cc = 100
gamm = 0.00001
svm_learner = svm.SVC(C = cc, gamma = gamm, probability = True)
print("C: ", cc)
print("gamma: ", gamm)
svm_learner.fit(X_train, Y_train)

Yhat = svm_learner.predict(X_valid)
Ythat = svm_learner.predict(X_train)

predictions = svm_learner.predict_proba(X_test)

print('Training Accuracy', np.mean (Ythat == Y_train))
print('Validation Accuracy', np.mean(Yhat == Y_valid))

print(predictions)

np.savetxt('SVM_predict.txt', np.vstack( (np.arange(len(predictions)), predictions[:,1]) ).T, 
            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

#C value determines how willing you are to misclassify data when
#deciding on the margin (higher the C, the less incorrect points are allowed within the margin)

#its said that a linear function aka no kernel can help prevent overfitting
# where a non linear function normally would, but that never finished running

#small gamma gives low bias and high variance while large gamma will give
#higher bias and low variance
