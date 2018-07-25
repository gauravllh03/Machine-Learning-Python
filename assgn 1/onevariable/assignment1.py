import numpy as np 
import matplotlib.pyplot as plt
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]

#PLOT INITIAL DATA
"""
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
plt.show()
"""

#COST FUNCTION FOR 1 VARIABLE
def computeCost(X, y, theta):
    m = y.size
    J = 0
    
    h = X.dot(theta)
    
    J = 1/(2*m)*np.sum(np.square(h-y))
    
    return(J)    
theta=np.zeros((X.shape[1],1))
print("Cost for one variable=",computeCost(X,y,theta))

#GRADIENT DESCENT FOR ONE VARIABLE
def gradientDescent(X, y, theta, alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)
fittheta,cost_j=gradientDescent(X,y,theta)
print("FITTED PARAMETERS FOR ONE VARIABLE=",fittheta.ravel())

# PLOT FOR GRADIENT VS NO. OF ITERATIONS
"""
plt.plot(cost_j,'r')
plt.xlabel('no. of iterations')
plt.ylabel('Cost function')
plt.title('GRADIENT DESCENT')
plt.show()
"""


#PLOT PREDICTION
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(X[:,1], X.dot(fittheta))
plt.show()




#python assignment1.py


