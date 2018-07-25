import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load MATLAB files
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression
data = loadmat('ex3data1.mat')
weights = loadmat('ex3weights.mat')
y = data['y']
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]
sample = np.random.choice(X.shape[0], 20)
initial_theta = np.zeros((X.shape[1],1))
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
def lrcostFunctionReg(theta,lamb, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1*(1/m)*(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h))) + (lamb/(2*m))*np.sum(np.square(theta[1:]))
    return(J)
print(lrcostFunctionReg(initial_theta,0,X,y));
def lrgradientReg(theta,lamb, X,y):
    m = y.size
    theta=np.reshape(theta,(X.shape[1],1))
    h = sigmoid(X.dot(theta))
    
    grad = (1/m)*X.T.dot(h-y) + (lamb/m)*theta
    grad[0]-=lamb*theta[0]/m
        
    return(grad.flatten())
print(lrgradientReg(initial_theta,0,X,y))

def oneVsAll(X, y, n_labels, lamb):
    initial_theta = np.zeros((X.shape[1],1))  # 401x1
    all_theta = np.zeros((n_labels, X.shape[1])) #10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args=(lamb, X , (y == c)*1), method=None,
                       jac=lrgradientReg, options={'maxiter':50})
        all_theta[c-1,:] = res.x
    return(all_theta)
all=oneVsAll(X,y,10,0)


def predictOneVsAll(all_theta,X):
    h = sigmoid(X.dot(all_theta.T))
    ind=np.argmax(h.T,axis=0)+1
    pred=ind.T
    return pred
    # Adding one because Python uses zero based indexing for the 10 columns (0-9),
    # while the 10 classes are numbered from 1 to 10.
    #return(np.argmax(probs, axis=1)+1)

pred=predictOneVsAll(all,X)
print(pred)

print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))

