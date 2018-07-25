import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
from sklearn.preprocessing import PolynomialFeatures


data = np.loadtxt('ex2data1.txt', delimiter=',')
X = np.c_[np.ones(data.shape[0]),data[:,0:2]]
y = np.c_[data[:,2]]

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
theta=np.zeros((X.shape[1],1))
print(theta)
def costFunction(theta,X,y):
	m=y.size
	j=0
	h=sigmoid(X.dot(theta))
	j=-1/m*(y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h)))
	return(j)
def gradient(theta, X, y):
    m = y.size
    theta=np.reshape(theta,(X.shape[1],1))
    h = sigmoid(X.dot(theta))
    
    grad =(1/m)*X.T.dot(h-y)

    return(grad.flatten())
initial_theta=theta
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)

res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})
print("solved theta=",res.x)

#PLOT INITIAL DATA
def plotData(X,y):
	pos=np.where(y==1)[0]
	neg=np.where(y==0)[0]
	plt.scatter(X[pos,1],X[pos,2],s=30,c='r',marker='+')
	plt.scatter(X[neg,1],X[neg,2],s=30,c='b',marker='o')


#PLOT LINEAR DCISION BOUNDARY	
def DecisionBoundary(X,y,theta):
	boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
	boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
	plotData(X,y)
	plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
	plt.legend()
	plt.show()

#PREDICTION
def makePrediction(theta, X):
    return sigmoid(X.dot(theta)) >= 0.5

#Compute the percentage of samples I got correct:
pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
pos_correct = float(np.sum(makePrediction(res.x,pos)))
neg_correct = float(np.sum(np.invert(makePrediction(res.x,neg))))
tot = len(pos)+len(neg)
prcnt_correct = float(pos_correct+neg_correct)/tot
print ("Fraction of training samples correctly predicted: %f." % prcnt_correct)

#python logisticregression.py