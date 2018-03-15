import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

iris = np.genfromtxt("C:\Python35\CS178\Lab3\data\iris.txt",delimiter=None)

X, Y = iris[:,0:2], iris[:,-1]  #get first two features & targett
X, Y = ml.shuffleData(X,Y)
X,_  = ml.transforms.rescale(X)

XA, YA = X[Y<2,:], Y[Y<2]       #get class 0 vs 1 Y[Y<2] = Y[0]
XB, YB = X[Y>0,:], Y[Y>0]       #get clsas 1 vs 2 Y[Y>0] = Y[1]

colors = ['b','r']
for c in np.unique(YA):
    plt.plot(XA[YA==c, 0], XA[YA==c, 1], 'o', color = colors[int(c)])
plt.title("Linearly Separable")
#plt.show()


for c in np.unique(YB):
    plt.plot(XB[YB==c, 0], XB[YB==c, 1], 'o', color = colors[int(c)-1])
plt.title("Not Linearly Separable")
#plt.show()

from logisticClassify2 import *

# data A
learnerA = logisticClassify2(); # create "blank" learner
learnerA.classes = np.unique(YA) # define class labels using YA or YB

wts = np.array([0.5,1,-.25]);  # set theta0, theta1, and theta2
learnerA.theta = wts; # set the learner's parameters

learnerA.plotBoundary(XA,YA)
#plt.show()


# data B
learnerB = logisticClassify2(); # create "blank" learner
learnerB.classes = np.unique(YB) # define class labels using YA or YB

wts = np.array([0.5,1,-.25]);  # set theta0, theta1, and theta2
learnerB.theta = wts; # set the learner's parameters


learnerB.plotBoundary(XB,YB)
#plt.show()


from sklearn.metrics import mean_squared_error


YAhat = learnerA.predict(XA)

mse = mean_squared_error(YA, YAhat)
print("Error rate on data set A: {}\n".format(mse))


YBhat = learnerB.predict(XB)

mseB = mean_squared_error(YB, YBhat)
print("Error rate on data set B: {}".format(mseB))


#ml.plotClassify2D(learnerA, XA, YAhat)
#plt.show()

#ml.plotClassify2D(learnerB, XB, YBhat)
#plt.show()


learnerA.train(XA,YA)
#plt.show()

learnerB.train(XB,YB)
#plt.show()

ml.plotClassify2D(learnerA, XA, YA)
plt.show()

ml.plotClassify2D(learnerB, XB, YB)
plt.show()

#plt.draw()


