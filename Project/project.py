import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

import sklearn.cross_validation
import sklearn.decomposition
import sklearn.model_selection
import sklearn.metrics


# import X training points with 15 features and Y training points
X_data = np.genfromtxt("C:\Python35\CS178\Project\X_train.txt",delimiter=None)
Y_data = np.genfromtxt("C:\Python35\CS178\Project\Y_train.txt",delimiter=None)

# import X testing points
X_test = np.genfromtxt("C:\Python35\CS178\Project\X_test.txt",delimiter=None)


from sklearn.model_selection import train_test_split
import sklearn.neighbors

# create K-nearest neighbor learner
knn_learner = ml.knn.knnClassify()

print(knn_learner)

# Different K values for nearest points
nearest = [1, 3, 5, 15, 55, 105]



# Subsampling a smaller part of the data
X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)

parameters = {'n_neighbors': nearest}

knearest = sklearn.neighbors.KNeighborsClassifier()

clf = sklearn.model_selection.GridSearchCV(knearest, parameters, cv = 10)

clf.fit(X_train, Y_train)

dimensions = [1,2,3,4,5,6,7,8,9,10]

accuracy = []
params = []


Test = X_test.shape[0]

predict = np.zeros((Test,2))


for d in dimensions:
    svd = sklearn.decomposition.TruncatedSVD(n_components = d)

    X_fit = svd.fit_transform(X_train)
    X_fit_atest = svd.transform(X_valid)

    clf.fit(X_fit, Y_train)

    Kfolds = sklearn.cross_validation.KFold(X_fit_atest.shape[0], n_folds = 10)
    scores = []

    for i,j in Kfolds:
        test_set = X_fit_atest[j]
        test_labels = Y_valid[j]
        scores.append(sklearn.metrics.accuracy_score(test_labels, clf.predict(test_set)))
        
    accuracy.append(scores)
    params.append(clf.best_params_['n_neighbors'])

    test = svd.transform(X_test)
    predict += clf.predict_proba(test)

print(np.mean(accuracy))
predict = predict/10
print(predict)

np.savetxt('1A-KNN.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T, 
           '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

# print(clf.best_params_)

##
##Test = X_test.shape[0]
##
##accuracy = []
##
##predict = np.zeros((Test,2))
##
####nFolds = 10;
####for iFold in range(nFolds):
####    
####    Xtr, Xva, Ytr, Yva = ml.crossValidate(X_data, Y_data, nFolds, iFold)
##
##Xtr, Xva, Ytr, Yva = train_test_split(X_data, Y_data, test_size=0.80, random_state = 42)
##
##    
##knn_learner = ml.knn.knnClassify()
##        
##    # best k-value is 5?
##knn_learner.train(Xtr, Ytr, 5) 
##        
##Yhat = knn_learner.predict(Xva)
##    
##    #submit.append(knn_learner.predict(X_test))
##    
###     fp,tp,tn = knn_learner.roc(Xva, Yva)
##    
###     plt.plot(fp,tp)
###     plt.show()
##    
##accuracy.append(knn_learner.auc(Xva, Yva))
##    
##predict += knn_learner.predictSoft(X_test)
##    
##print(predict)
###predict = predict/10
##    
##print(np.mean(accuracy))
##
###full_ensemble.append(predict)
##
##
##np.savetxt('1A-KNN.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T, 
##           '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')
##

##
##import sklearn.model_selection
##import sklearn.neighbors
##
### create K-nearest neighbor learner
##knn_learner = ml.knn.knnClassify()
##
##print(knn_learner)
##
### Different K values for nearest points
##nearest = [1, 3, 5, 15, 55, 105]
##
##
##
### Subsampling a smaller part of the data
##X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)
##
##
##parameters = {'n_neighbors': nearest}
##
##knearest = sklearn.neighbors.KNeighborsClassifier()
##
##clf = sklearn.model_selection.GridSearchCV(knearest, parameters, cv = 10)
##
##
##clf.fit(X_train, Y_train)
##
##
##print(clf.best_params_)





##
##from sklearn.ensemble import AdaBoostClassifier
##from sklearn.tree import DecisionTreeClassifier
##from sklearn.naive_bayes import GaussianNB
##

##
##X_train = X_data[0:10000]
##Y_train = Y_data[0:10000]
##
##X_valid = X_data[10000:20000]
##Y_valid = Y_data[10000:20000]



##decision_tree = DecisionTreeClassifier()
##
###AUC = []
##
##points = []
##
##ensemble = [None] * 25
##
##estimators = [50,100,150,200,250,300,350,400]
##
##for each in estimators:
##
##    AUC = []
##    
##    for i in range(0, 25):
##        
##        indices = np.floor(10000 * np.random.rand(100000)).astype(int)    # random combo of M rows
##        
##        X_train, Y_train = X_data[indices,:], Y_data[indices]            # X and Y indices of those rows
##        
##        
##        ensemble[i] = AdaBoostClassifier(n_estimators = each, 
##                                         base_estimator = DecisionTreeClassifier(max_depth = 1),
##                                         learning_rate = 1)
##        
##        ensemble[i].fit(X_train, Y_train)
##
##        ensemble[i].predict(X_valid)
##
##        AUC.append(np.mean(ensemble[i].predict(X_valid) == Y_valid))
##    
##    points.append(np.mean(AUC))
##
##plt.plot(estimators, points)
##
##plt.show()

# for i in range(10,200):
    
#     boosting_trees = AdaBoostClassifier(n_estimators = i, 
#                                         base_estimator = DecisionTreeClassifier(max_depth = 1), 
#                                         learning_rate = 1)
    
    
    
# for i in range(0,25):
#     print(AUC[i])

#print(np.mean(AUC))

#print("AdaBoost AUC: {}".format(result))

# create K-nearest neighbor learner
##knn_learner = ml.knn.knnClassify()
####print(knn_learner)
##
##
##
### Different K values for nearest points
##nearest = [1, 5, 15, 55]
##
##
### Subsampling a smaller part of the data
##X_train = X_data[0:10000]
##Y_train = Y_data[0:10000]
##
##X_valid = X_data[10000:20000]
##Y_valid = Y_data[10000:20000]
##
##
##print(np.zeros(10))



##clf = ml.nnet.nnetClassify()
##
##clf.train(X_train, Y_train, init = 'random')
##
##
##
##
##Yhat = clf.predictSoft(X_valid)
##
##
##fp, tp, tn = nnet.roc(X_valid, Y_valid)
##plt.plot(fp,tp)
##plt.show()
##
##
##auc = clf.auc(X_valid, Y_valid)
##
##print(auc)


##from sklearn.neural_network import MLPClassifier
##
##
##neural_network = MLPClassifier(solver = 'adam', alpha = 1e-5,
##                               hidden_layer_sizes = (5,2),
##                               random_state = 1)
##
##neural_network.fit(X_train, Y_train)
##
##Yhat = neural_network.predict_proba(X_valid)
##
##
##print(np.mean(neural_network.predict(X_valid) == Y_valid))
##
##
##print(Yhat)









###print(X_train[0])
##
##for k in nearest:
##
##    knn_learner.train(X_train, Y_train, k)
##
##    Yhat = knn_learner.predict(X_valid)
##
##    Y_test = knn_learner.predictSoft(X_valid)
##
###     # Can only compare 2 features
###     ml.plotClassify2D( knn_learner, X_train, Y_train );
###     plt.show()
##
##    false_pos, true_pos, true_neg = knn_learner.roc(X_valid, Y_valid)
##
##    areaUnderCurve = knn_learner.auc(X_valid, Y_valid)
##
##
##    print("K value = {:<2}      auc: {}".format(k, areaUnderCurve))
##
##    plt.plot(false_pos, true_pos)
##    plt.show()
##    
##
##trErr = []
##valErr = []
##k_values = [1,3, 5, 15, 25, 55, 105, 205]
##
##for i in k_values:
##    knn_learner.K = i
##    trErr.append(knn_learner.err(X_train,Y_train))
##    valErr.append(knn_learner.err(X_valid, Y_valid))
##
##
##plt.semilogx(k_values, trErr, 'r-', nearest, valErr, 'g-')
##plt.legend(['Training','Validation'])
##plt.show()
##
###print("KNN AUC: {}".format(result))
