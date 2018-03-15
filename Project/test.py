
#### WORKING ####

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA

# import X training points with 15 features and Y training points
X_data = np.genfromtxt("C:\Python35\CS178\Project\X_train.txt",delimiter=None)
Y_data = np.genfromtxt("C:\Python35\CS178\Project\Y_train.txt",delimiter=None)

# import X testing points
X_test = np.genfromtxt("C:\Python35\CS178\Project\X_test.txt",delimiter=None)

X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)


##from sklearn.naive_bayes import GaussianNB
##gnb = GaussianNB()
##y_pred = gnb.fit(X_train, Y_train).predict(X_train)
##print("Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0],(Y_train != y_pred).sum()))



Test = X_test.shape[0]
predict = np.zeros((Test,2))

from sklearn.neural_network import MLPClassifier

""" solver = 'adam' for large data sets """


##average = []
##
##nFolds = 10;
##
##for iFold in range(nFolds):
##
##    Xtr, Xva, Ytr, Yva = ml.crossValidate(X_data, Y_data, nFolds, iFold)
##
##    neural_network = MLPClassifier(solver = 'adam', random_state = 0)
##
##    neural_network.fit(Xtr, Ytr)
##
##    predict += neural_network.predict_proba(X_test)
##
##    average.append(np.mean(neural_network.predict(Xva) == Yva))
##
##    #print(Yhat)
##
##print(np.mean(average))
##predict = predict/10
##print(predict)
##
##np.savetxt('NNET.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T,
##           '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')


##
###### RANDOM FOREST OF DECISION TREES ####
##
### dimensions of X_test
##m,n = X_test.shape
##
##Test = X_test.shape[0]
##
### ensemble of classifiers
##bags = 50
##full_ensemble = [None] * bags
##Area_Under_Curve = []
##
##
##for i in range(0, bags):
##    
##    indices = np.floor(m * np.random.rand(m)).astype(int)    # random combination of 100k rows
##    
##    Xi, Yi = X_data[indices,:], Y_data[indices]              # X and Y indices of those rows
##    
##    # put the learners in the ensemble
##    full_ensemble[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth = 20, minLeaf = 4, nFeatures = 10) 
##    
##
### space for predictions from each model
##predict = np.zeros((Test,2))
##
##
##for i in range(0,bags):
##    
##    # soft predict on each learner in the bag
##    predict += full_ensemble[i].predictSoft(X_test)
##    
##    # get the auc of each learner in the bag
##    Area_Under_Curve.append(full_ensemble[i].auc(X_data, Y_data))
##
### average predictions    
##predict = predict/bags
##
### average auc
##result = np.mean(Area_Under_Curve)
##print("Random Forest AUC: {}".format(result))
##
### np.savetxt('Yhat_ensemble.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T, 
###            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# adaptive boost bayes classifier

bayes_learner = ml.bayes.gaussClassify()

bayes_learner.train(X_train, Y_train)

#Yhat = bayes_learner.predictSoft(X_valid)



#GaussianNaiveBayes = GaussianNB()

decision_tree = DecisionTreeClassifier()

AUC = []

Test = X_test.shape[0]
ultraboost = np.zeros((Test,2))

ensemble = [None] * 25


for i in range(0, 25):
    
    Xtr, Xva, Ytr, Yva = train_test_split(X_data, Y_data, test_size=0.80, random_state = 42)

    
    ensemble[i] = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 15))
    
    ensemble[i].fit(Xtr, Ytr)


    AUC.append(np.mean(ensemble[i].predict(Xva) == Yva))
    
    ultraboost += ensemble[i].predict_proba(X_test)


# for i in range(10,200):
    
#     boosting_trees = AdaBoostClassifier(n_estimators = i, 
#                                         base_estimator = DecisionTreeClassifier(max_depth = 1), 
#                                         learning_rate = 1)
    

ultraboost = ultraboost/25
print(ultraboost)

np.savetxt('Ultraboost.txt', np.vstack( (np.arange(len(ultraboost)), ultraboost[:,1]) ).T, 
           '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')
    
# for i in range(0,25):
#     print(AUC[i])

print(np.mean(AUC))

#print("AdaBoost AUC: {}".format(result))









