import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = np.loadtxt('ex1data2.txt', delimiter=',')

#INITIAL DATA PLOT

X = np.c_[np.ones(data.shape[0]),data[:,0:2]]
y = np.c_[data[:,2]]

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1],X[:,2],y, c='r', marker='o')

ax.set_xlabel('size')
ax.set_ylabel('no. of bedrooms')
ax.set_zlabel('price of house')

plt.show()
"""
#FEATURE NORMALIZATION
def featureNormalize(X):
	Xnorm=X.copy()
	means=np.mean(Xnorm,axis=0)
	Xnorm[:,1:]=Xnorm[:,1:]-means[1:]
	stds=np.std(Xnorm,axis=0,ddof=1)#ddof->delta degrees of freedom
	Xnorm[:,1:]=Xnorm[:,1:] / stds[1:]
	return Xnorm

Xnorm=featureNormalize(X)
#COST FUNCTION FOR MULTIPLE VARIABLES
def computeCost(X, y, theta):
    m = y.size
    J = 0
    
    h = X.dot(theta)
    
    J = 1/(2*m)*np.sum(np.square(h-y))
    
    return(J)    
theta=np.zeros((Xnorm.shape[1],1))
print("Cost for multiple variable=",computeCost(Xnorm,y,theta))

#GRADIENT DESCENT FOR MULTIPLE VARIABLES
def gradientDescent(X, y, theta, alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)
fittheta,cost_j=gradientDescent(Xnorm,y,theta)
print("FITTED PARAMETERS FOR ONE VARIABLE=",fittheta.ravel())

# PLOT FOR GRADIENT VS NO. OF ITERATIONS
"""
plt.plot(cost_j,'r')
plt.xlabel('no. of iterations')
plt.ylabel('Cost function')
plt.title('GRADIENT DESCENT')
plt.show()
"""
#python multivar.py