import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC


def plotData(X, y):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    
    plt.scatter(X[pos,0], X[pos,1], s=60, c='k', marker='+', linewidths=1)
    plt.scatter(X[neg,0], X[neg,1], s=60, c='y', marker='o', linewidths=1)
    plt.show()

def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plotData(X, y)
    #plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

 

 #Dataset1 LINEAR DB.remove """ to implement this.
"""
data1 = loadmat('ex6data1.mat')   
X1=data1['X']
y1=data1['y']
#plotData(X1,y1)

#TRAINING SVM FOR LINEAR PLOT with C=1
clf = SVC(C=1.0, kernel='linear')
clf.fit(X1, y1.ravel())
plot_svc(clf, X1, y1)



#TRAINING SVM FOR LINEAR PLOT with C=100
clf = SVC(C=100, kernel='linear')
clf.fit(X1, y1.ravel())
plot_svc(clf, X1, y1)

"""





#SVM FOR NON-LINEAR PLOTS USING GAUSSIAN KERNEL
def gaussianKernel(x1, x2, sigma=2):
    norm = (x1-x2).T.dot(x1-x2)
    return(np.exp(-norm/(2*sigma**2)))
#TESTING GAUSSIAN KERNEL
"""
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

print(gaussianKernel(x1, x2, sigma))
"""

#DATASET2 FOR NON LINEAR PLOT
"""
data2 = loadmat('ex6data2.mat')
y2 = data2['y']
X2 = data2['X']

#plotData(X2, y2)

#TRAINING SVM FOR NON LINEAR
clf2 = SVC(C=50, kernel='rbf', gamma=10)
clf2.fit(X2, y2.ravel())
plot_svc(clf2, X2, y2)
"""


#DATASET3 FOR NON LINEAR PLOT

data3 = loadmat('ex6data3.mat')
y3 = data3['y']
X3 = data3['X']

#plotData(X3, y3)

clf3 = SVC(C=1.0, kernel='poly',degree=3,gamma=10)
clf3.fit(X3, y3.ravel())
plot_svc(clf3, X3, y3)

#python svm.py