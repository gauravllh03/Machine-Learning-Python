import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression

# load MATLAB files
from scipy.io import loadmat
data = loadmat('ex4data1.mat')
y=data['y']
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]
weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
params = np.r_[theta1.ravel(), theta2.ravel()]
m=X.shape[0]
input_layer_size=400
hidden_layer_size=25
num_labels=10
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
def sigmoidGradient(z):
    return(sigmoid(z)*(1-sigmoid(z)))
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y,lamb):
    
    # When comparing to Octave code note that Python uses zero-indexed arrays.
    # But because Numpy indexing does not include the right side, the code is the same anyway.
    theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,(hidden_layer_size+1))

    m = X.shape[0]
    y_matrix = pd.get_dummies(y.ravel()).as_matrix() 
    # Cost
    a1 = X # 5000x401
        
    z2 = a1.dot(theta1.T) # 5000*26
    #a2 = np.c_[np.ones((X.shape[0],1)),sigmoid(z2)] # 5000x26 
    a2=sigmoid(z2)
    a2=np.c_[np.ones((a2.shape[0],1)),a2]
    z3 = a2.dot(theta2.T) # 5000*10
    a3 = sigmoid(z3)
    h=a3 # 5000*10
    J = -1*(1/m)*np.sum(((y_matrix)*np.log(h)+(1-y_matrix)*np.log(1-h))) + \
        (lamb/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))
    



    #gradient calculations
    d3 = a3 - y_matrix # 5000x10
    d2 = theta2[:,1:].T.dot(d3.T)*sigmoidGradient(z2.T) # 25x10 *10x5000 * 25x5000 = 25x5000
    
    delta1 = d2.dot(a1) # 25x5000 * 5000x401 = 25x401
    delta2 = d3.T.dot(a2) # 10x5000 *5000x26 = 10x26
    
    theta1_ = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]
    theta2_ = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]
    
    theta1_grad = delta1/m + (theta1_*lamb)/m
    theta2_grad = delta2/m + (theta2_*lamb)/m
    
    grad=np.r_[theta1_grad.ravel(), theta2_grad.ravel()]
    return J,grad




#J,grad=nnCostFunction(params, 400,25,10, X, y ,0)

def randInitializeWeights(L_in,L_out):
    """
    Randomly initalize the weights of a layer with L_in incoming
    connections and L_out outgoing connections. Avoids symmetry
    problems when training the neural network.
    """
    randWeights = np.random.uniform(low=-.12,high=.12,
                                    size=(L_in,L_out))
    return randWeights

t1=randInitializeWeights(input_layer_size+1,hidden_layer_size)
t2=randInitializeWeights(hidden_layer_size+1,num_labels)
initial_theta=np.r_[t1.ravel(),t2.ravel()]
print("initial_theta=",initial_theta)
initial_cost, g = nnCostFunction(initial_theta,input_layer_size,
                                 hidden_layer_size,num_labels,X,y,0)

#print("The initial cost after random initialization: ", initial_cost)

reg_param=3
def reduced_cost_func(p):
    
    return nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,
                          X,y,reg_param)

results = minimize(reduced_cost_func,
                   initial_theta,
                   method="CG",
                   jac=True,
                   options={'maxiter':50, "disp":True})
fitted_params=results.x
print("Fitted parameters =",fitted_params.shape)
fitted_theta1 = fitted_params[:(hidden_layer_size * 
             (input_layer_size + 1))].reshape((hidden_layer_size, 
                                       input_layer_size + 1))

fitted_theta2 = fitted_params[-((hidden_layer_size + 1) * 
                      num_labels):].reshape((num_labels,
                                   hidden_layer_size + 1)) 

print("fittedtheta1=",fitted_theta1.shape)
print("fittedtheta2=",fitted_theta2.shape)


def prediction(T1,T2,X):
    p=np.zeros((X.shape[0],1))
    a2=sigmoid(X.dot(T1.T))
    a2=np.c_[np.ones((a2.shape[0],1)),a2]
    h=sigmoid(a2.dot(T2.T))
    ind=np.argmax(h.T,axis=0)+1
    pred=ind.T
    return pred
prediction=prediction(fitted_theta1,fitted_theta2,X)
print('Training set accuracy: {} %'.format(np.mean(prediction == y.ravel())*100))

