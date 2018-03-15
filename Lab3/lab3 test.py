import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
#from sklearn.metrics import mean_squared_error
from logisticClassify2 import *


iris = np.genfromtxt("C:\Python35\CS178\Lab3\data\iris.txt",delimiter=None)

X, Y = iris[:,0:2], iris[:,-1]  #get first two features & targett
X, Y = ml.shuffleData(X,Y)
X,_  = ml.transforms.rescale(X)

XA, YA = X[Y<2,:], Y[Y<2]       #get class 0 vs 1 Y[Y<2] = Y[0]
XB, YB = X[Y>0,:], Y[Y>0]       #get clsas 1 vs 2 Y[Y>0] = Y[1]
#print(XA,YA)
##
##colors = ['b','r']
##for c in np.unique(YA):
##    plt.plot(XA[YA==c, 0], XA[YA==c, 1], 'o', color = colors[int(c)])
##plt.title("Linearly Separable")
##plt.show()
##
##
##for c in np.unique(YB):
##    plt.plot(XB[YB==c, 0], XB[YB==c, 1], 'o', color = colors[int(c)-1])
##plt.title("Not Linearly Separable")
##plt.show()
##
##from logisticClassify2 import *
##
##learner = logisticClassify2(); # create "blank" learner
##learner.classes = np.unique(YA) # define class labels using YA or YB
##
##wts = np.array([0.5,1,-.25]); # TODO: fill in values
##learner.theta = wts; # set the learner's parameters
##
##
##learner.plotBoundary(XA,YA)

learner = logisticClassify2(); # create "blank" learner
learner.classes = np.unique(YA) # define class labels using YA or YB

wts = np.array([0.5,1,-.25]);  # set theta0, theta1, and theta2
learner.theta = wts; # set the learner's parameters

learner.predict(XA)


classes = [0,1]
Yhat = np.zeros(len(XA))
theta = np.array([1,2,3,7,8,9])

r = np.zeros(len(XA))

thing = []

for i in range(0,len(XA)):
    r[i] = theta[0] + (X[i][0]*theta[1]) + (X[i][1]*theta[2])

    if r[i] > 0 :
        Yhat[i] = classes[1]
    else:
        Yhat[i] = classes[0]
#    print(YA[i])
    #thing.append(YA[i])
    

##print(r)
#print(Yhat)


print(theta.T)
print(theta[1:].T)


