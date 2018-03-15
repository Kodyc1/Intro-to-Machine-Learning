import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

lst = list(range(0,10))

print(lst[0:2])

##classifiers = [None] * 25
##
##print(classifiers)
##for i in range(2,13):
##    print(2**i)
##
##M = 10
##
##for i in range(0,25):
##
##    indices = np.floor(10 * np.random.rand(10)).astype(int)
##
##    print(indices)
    

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml


X_data = np.genfromtxt("C:\Python35\CS178\Lab4\X_train.txt",delimiter=None)
Y_data = np.genfromtxt("C:\Python35\CS178\Lab4\Y_train.txt",delimiter=None)

# First 10,000 points are for training, second 10,000 are for validation
X_train = X_data[0:10000]
X_valid = X_data[10001:20000]

Y_train = Y_data[0:10000]
Y_valid = Y_data[10001:20000]


from sklearn.metrics import mean_squared_error

learner = ml.dtree.treeClassify(X_train, Y_train, maxDepth = 50)


# Calculate predicted training and validation error rates
Yhat_Train = learner.predict(X_train)

mse = mean_squared_error(Y_train, Yhat_Train)

Yhat_Validation = learner.predict(X_valid)

mse_valid = mean_squared_error(Y_valid, Yhat_Validation)


print("Training MSE of Depth 50: {}      Validation MSE of Depth 50: {}\n".format(mse,mse_valid))

valid = []
for i in range(0,16):
    
    small_learner = ml.dtree.treeClassify(X_train, Y_train, maxDepth = i)
    
    Yhat_tr = small_learner.predict(X_train)
    
    tr_mse = mean_squared_error(Y_train, Yhat_tr)
    
    Yhat_val = small_learner.predict(X_valid)

    val_mse = mean_squared_error(Y_valid, Yhat_val)
    
    valid.append(val_mse)

    print("Training MSE of Depth {:<2}: {:<6}      Validation MSE of Depth {:<2}: {}".format(i, tr_mse, i, val_mse))

    
plt.plot(list(range(0,16)), valid)
plt.show()


error = []
for i in range(2,13):
    
    min_learner = ml.dtree.treeClassify(X_train,Y_train, maxDepth = 50, minLeaf = 2**i)
    
    Yhat_tr = min_learner.predict(X_train)
    
    tr_mse = mean_squared_error(Y_train, Yhat_tr)
    
    Yhat_val = min_learner.predict(X_valid)

    val_mse = mean_squared_error(Y_valid, Yhat_val)
    
    error.append(val_mse)

    print("Training MSE with minLeaf {:<4}: {:<6}    Validation MSE with minLeaf {:<4}: {}".format(2**i, tr_mse, 2**i, val_mse))
    
    
plt.plot(list(range(2,13)), error)
plt.show()


#roc member function

false_positive, true_positive, true_negative = learner.roc(X_valid, Y_valid)

plt.plot(false_positive, true_positive)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#auc member function

print("Area under the curve: {}".format(learner.auc(X_valid, Y_valid)))



X_test = np.genfromtxt("C:\Python35\CS178\Lab4\X_test.txt",delimiter=None)

best_learner = ml.dtree.treeClassify(X_data, Y_data, maxDepth = 50, minLeaf = 128)

Y_predictions = best_learner.predictSoft(X_test)


false_positive, true_positive, true_negative = best_learner.roc(X_valid, Y_valid)

plt.plot(false_positive, true_positive)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("auc: ", best_learner.auc(X_data, Y_data))

np.savetxt('Yhat_dtree.txt', np.vstack( (np.arange(len(Y_predictions)), Y_predictions[:,1]) ).T, 
           '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')


# Build ensemble members

M,N = X_train.shape

learners = [1, 5, 10, 15, 20, 25]



# ensemble of classifiers
ensemble = [None] * 25

for i in range(0, 25):
    #ml.bootstrapData(X_train, Y_train, n_boot = nBag)
    
    indices = np.floor(M * np.random.rand(M)).astype(int)    # random combo of M rows
    
    Xi, Yi = X_train[indices,:], Y_train[indices]                        # X and Y indices of those rows
    
    ensemble[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth = 15, minLeaf = 4, nFeatures = 8) # put the learners in the ensemble



    
    
# Compute prediction of the ensemble

training_err = []
validation_err = []

Test = X_test.shape[0]

# space for predictions from each model
predict = np.zeros((Test,25))

for i in range(0,25):

    predict[:,i] = ensemble[i].predict(X_test)

for bags in learners:

    tr_err = np.mean(Y_train != (np.mean(predict[:,:bags], axis=1) >0.5 ))
    
    training_err.append(tr_err)

    val_err = np.mean(Y_valid != (np.mean(predict[:,:bags], axis=1) >0.5 ))
    
    validation_err.append(val_err)

plt.plot(learners, training_err)
plt.plot(learners, validation_err)
plt.show()

