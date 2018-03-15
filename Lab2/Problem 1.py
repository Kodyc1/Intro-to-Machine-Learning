import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

# Linear Regression

data = np.genfromtxt("C:\Python35\CS178\Lab2\data\curve80.txt",delimiter=None)
# 1A Splitting 2 column data/curve80.txt into 75/25%
X = data[:,0]
X = X[:,np.newaxis]  # code expects (M,N), newaxis

Y = data[:,1]        
Xtr, Xte, Ytr, Yte = ml.splitData(X,Y,0.75) #split data 75 Training/25 Testing

print(data)


# 1B Creating linear regression predictor of y given x

lr = ml.linear.linearRegress( Xtr, Ytr ); # creating and training model
xs = np.linspace(0,10,200);               # sample large number of possible x values
xs = xs[:,np.newaxis]                     # force it to be an single column matrix
ys = lr.predict( xs );                    # make predictions on it

# Plot the training data along with your prediction function.
plt.plot(Xtr, Ytr,'o')
plt.plot(xs,ys)
##plt.show()

# Print the regression coefficicents (lr.theta)
### lr.theta???
plt.plot([0,1,3.3],[-2.8276,-2,0],c='g')
plt.show()
print(lr.theta)

# Check they match the plot


# Calculate and report the mean squared error for both data





### 1c fitting y = f(x) increasing order
##
##Xtr2 = np.zeros( (Xtr.shape[0], 2) )      # Mx2 matrix array to store features
##Xtr2[:,0] = Xtr[:,0]                      # place original X feature as X1
##Xtr2[:,1] = Xtr[:,0]**2                   # place x^2 feature as X2
##
##
### Create polynomial features up to degree; don't make it constant
##XtrP = ml.transforms.fpoly(Xtr, degree, bias=False)
##
##
### Rescale the data matrix so that the features have similar ranges / variance
##XtrP.params = ml.transforms.rescale(XtrP);
##
##
### Then we can train the model on the scaled feature matrix
##lr = ml.linear.linearRegress( XtrP, Ytr );
##
##
##
##XteP = ml.transforms.rescale( ml.transforms.fpoly(Xte, degree, false), params);
##
##

