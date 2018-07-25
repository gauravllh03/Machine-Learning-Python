import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import linalg

data = loadmat('ex7data1.mat')
X = data['X']
#plot initial data
#plt.plot(X[:,0], X[:,1], 'bo')
#plt.show()

#Function to normalize data
def featureNormalize(X):
    
    # compute mean and standard deviation
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma  


#Function to return eigen vectors
def pca(X):
    
    m, n = X.shape
    
    # compute covariance matrix
    Sigma = np.dot(X.T, X) / m
    
    # SVD
    u, s, v = np.linalg.svd(Sigma)
    
    return u, s

#Function to draw line from 1 point t0 another
def drawLine(p1, p2, *args, **kwargs):
    
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)




X_norm, mu, sigma = featureNormalize(X)

# perform PCA
U, S = pca(X_norm)
#plot eigen vectors
"""
plt.figure(figsize=(5,5))
plt.plot(X[:,0], X[:,1], 'bo')
drawLine(mu, mu+1.5*S[0]*U[:,0], lw=2, color='k')
drawLine(mu, mu+1.5*S[1]*U[:,1], lw=2, color='k')
plt.xlim(0.5, 7)
plt.show()
"""

#projecting data
def projectData(X, U, K):
    
    return np.dot(X, U[:,:K])

#recovering data
def recoverData(Z, U, K):
    
    return np.dot(Z, U[:,:K].T)

K = 1
Z = projectData(X_norm, U, 1)
X_rec = recoverData(Z, U, K)

#plotting projections
"""
plt.figure(figsize=(5,5))
plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
plt.plot(X_rec[:,0], X_rec[:,1], 'ro')
for ii in range(X_norm.shape[0]):
    drawLine(X_norm[ii,:], X_rec[ii,:], '--k')
plt.xlim(-4,3)
plt.ylim(-4,3)
plt.show()
"""






#PCA with face images

data2=loadmat('ex7faces.mat')
Xface=data2['X']


#DISPLAY DATA FUNCTION (2D MATRIX)

def displayData(X, nrows=10, ncols=10):
    
    # set up array
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols,
                              figsize=(nrows,ncols))
    
    nblock = int(np.sqrt(X.shape[1]))
    
    # loop over randomly drawn numbers
    ct = 0
    for ii in range(nrows):
        for jj in range(ncols):
            #ind = np.random.randint(X.shape[0])
            tmp = X[ct,:].reshape(nblock, nblock, order='F')
            axarr[ii,jj].imshow(tmp, cmap='gray')
            plt.setp(axarr[ii,jj].get_xticklabels(), visible=False)
            plt.setp(axarr[ii,jj].get_yticklabels(), visible=False)
            plt.minorticks_off()
            ct += 1
    
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.show()


#plotting original faces and faces recovered after pca dimensionality reduction   
#displayData(Xface)

X_norm_face, mu_f, sigma_f = featureNormalize(Xface)

# run PCA
U_face, S_face = pca(X_norm_face)
K_f = 100
Z_f = projectData(X_norm_face, U_face, K_f)
X_rec_f=recoverData(Z_f, U_face, K_f)


#plotting recovered faces

displayData(X_rec_f*sigma_f + mu_f)
plt.suptitle('Recovered')

