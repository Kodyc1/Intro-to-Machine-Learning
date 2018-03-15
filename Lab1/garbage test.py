# CS178 LAB 1 WINTER 2017
# KODY CHEUNG 85737824

import numpy as np
import matplotlib.pyplot as plt


###load text file
##iris = np.genfromtxt("C:\Python34\CS178\data\iris.txt",delimiter=None)
##
###print(iris)
###targeting last column
##Y = iris[:,-1]
###featuring other columns
##X = iris[:,0:-1]
##
###print(X)
##
##
####### 1A #####
##
##X.shape[1] #number of features
###print(X.shape[1])
##
##X.shape[0] #number of data points
###print(X.shape[0])
##
##
####### 1B #####
##plt.hist(X[:,0],label="Feature 1",rwidth=0.8)
##plt.hist(X[:,1],label="Feature 2",rwidth=0.8)
##plt.hist(X[:,2],label="Feature 3",rwidth=0.5)
##plt.hist(X[:,3],label="Feature 4",rwidth=0.8)
##
##plt.legend()
##plt.show()
##
##
##
####### 1C #####
###(np.mean, np.std)
##mean_one = np.mean(X[:,0])
##std_one = np.std(X[:,0])
##print("Mean of Feature 1: {:32}".format(mean_one))
##print("Standard deviation of Feature 1: {:18}\n".format(std_one))
##
##mean_two = np.mean(X[:,1])
##std_two = np.std(X[:,1])
##print("Mean of Feature 2: {:32}".format(mean_two))
##print("Standard deviation of Feature 2: {:20}\n".format(std_two))
##
##mean_three = np.mean(X[:,2])
##std_three = np.std(X[:,2])
##print("Mean of Feature 3: {:33}".format(mean_three))
##print("Standard deviation of Feature 3: {:19}\n".format(std_three))
##
##mean_four = np.mean(X[:,3])
##std_four = np.std(X[:,3])
##print("Mean of Feature 4: {:33}".format(mean_four))
##print("Standard deviation of Feature 4: {:19}".format(std_four))
##
##
##
##
####### 1D #####
### for (1,2) (1,3) and (1,4)  plt.plot or plt.scatter
##
##colors = ['b','g','r']
##
##for c in np.unique(Y):
##    plt.plot(X[Y==c, 0], X[Y==c,1], 'o', label = "Class value: {}".format(c), color = colors[int(c)])
##    
##plt.title('(1,2)')
##ax = plt.gca()
##ax.set_ylim([2.0,5])
##plt.legend(loc='best')
##plt.show()
##
##for c in np.unique(Y):
##    plt.plot(X[Y==c, 0], X[Y==c,2], 'o', label = "Class value: {}".format(c), color = colors[int(c)])
##
##plt.title('(1,3)')
##ax = plt.gca()
##ax.set_ylim([1,8])
##plt.legend(loc='best')
##plt.show()
##
##for c in np.unique(Y):
##    plt.plot(X[Y==c, 0], X[Y==c,3], 'o',label = "Class value: {}".format(c),color = colors[int(c)])
##
##plt.title('(1,4)')
##ax = plt.gca()
##ax.set_ylim([0,3.5])
##plt.legend(loc='best')
##plt.show()


import mltools as ml

##### PROBLEM 2 #####
iris = np.genfromtxt("data/iris.txt", delimiter = None)

# Note: indexing with ":" indicates all values (in this case, all rows)

# indexing with a value ("0", "1", "-1", etc.) extracts only that one value (here, columns);
# indexing rows/columns with a range ("1:-1") extracts any row/column in that range.
Y = iris[:,-1]   # last column (0, 1, 2, 3, -1)
X = iris[:,0:2] # takes first 2 columns out of 5

print(Y)
print(X)

X,Y = ml.shuffleData(X,Y)  # Shuffles the ordered Iris data

# Xtr = 75% of X[0:2]
# Xva = 25% of X[0:2]
Xtr, Xva, Ytr, Yva = ml.splitData(X,Y, 0.75); # split it into 75/25 train/validation






##
##knn = ml.knn.knnClassify() #create object and train it
##knn.train(Xtr, Ytr, 1)     #where K is an integer, e.g. 1 for nearest neighbor prediction
##YvaHat = knn.predict(Xva)  #get estimates of y for each data point in Xva
##
##ml.plotClassify2D( knn, Xtr, Ytr ) # make 2D classification plot with data (Xtr,Ytr)
##plt.title("K = 1")
##plt.show()
##
##
##### Alternatively, the constructor provides a shortcut to "train":
####  knn = ml.knn.knnClassify( Xtr, Ytr, K );
####  YvaHat = predict( knn, Xva );
##
##knn = ml.knn.knnClassify( Xtr, Ytr, 5 );
##YvaHat = knn.predict(Xva);
##
##ml.plotClassify2D( knn, Xtr, Ytr )
##plt.title("K = 5")
##plt.show()
##
##
##knn = ml.knn.knnClassify( Xtr, Ytr, 10 );
##YvaHat = knn.predict(Xva);
##
##ml.plotClassify2D( knn, Xtr, Ytr ) 
##plt.title("K = 10")
##plt.show()
##
##
##knn = ml.knn.knnClassify( Xtr, Ytr, 50 );
##YvaHat = knn.predict(Xva);
##
##ml.plotClassify2D( knn, Xtr, Ytr )
##plt.title("K = 50")
##plt.show()
