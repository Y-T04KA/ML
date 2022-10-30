import scipy
from scipy import stats as sps

from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
matplotlib.style.use('ggplot')



#1
A=np.array([8.2, 2.8, 3.9, 2.7, 7.4, 3.4, 9.6, 7.4])
B=np.array([7.3, -2, -1.4, -3, 6.3, -2, 6.6, 4.2 ])
X=np.stack([A,B]).T
plt.scatter(A,B) 
for i in range(len(A)):
 plt.annotate(f'{i+1}', xy=(A[i]-0.1, B[i]+0.1))
plt.show()
kernel = np.ndarray(shape=(X.shape[0], X.shape[0]), dtype=float)
for i, xi in enumerate(X):
    for j, xj in enumerate(X):
        dx = xi - xj
        kernel[i][j] = np.round(np.linalg.norm(dx)**2,3)     
     
print("Kernel:")        
print(kernel)

#2
A=np.array([-86,16,-84,19,-74,-114,-76,-83])
B=np.array([115,-19,-223,19,-48,37,31,-73])
X1=np.stack([A,B])
X=np.stack([A,B]).T
plt.scatter(A,B)#это получается пункт 1
for i in range(len(A)):
 plt.annotate(f'{i+1}', xy=(A[i]-0.1, B[i]+0.1))
plt.xlabel("Array A")
plt.ylabel("Array B")
plt.show()
mean1=np.array([np.mean(A),np.mean(B)])
cm=np.cov(X)
print("X.mean:")#тут начинается 2
print(mean1)
print("Cov. matrix:")
print(cm)
Xcentered=(X[0]-A.mean(),X[1]-B.mean())
Xcc=np.cov(Xcentered)
print("Cov. matrix for centered:")
print(Xcc) #тут 2 заканчивается
print("eig:")#3
w,v=np.linalg.eig(Xcc)
print(w)#chisla
print(v)#vector
#вот 4 я не понял, почитал методичку и всё равно не понял
mu = np.zeros(2)
C = Xcc
data=X
v, W_true = np.linalg.eig(C)
plt.scatter(data[:,0], data[:,1])
test=(W_true[1,0]/W_true[1,1])*data[:,0]

for i in range(len(A)):
 plt.annotate(f'{i+1}', xy=(A[i]-0.1, test[i]-0.1))
plt.plot(data[:,0], (W_true[1,0]/W_true[1,1])*data[:,0], "go")

model=PCA(n_components=1)
model.fit(data)
W_pca=model.components_
test2=(W_pca[0,0]/W_pca[0,1])*data[:,0]
plt.plot(data[:,0], test2 , "co")
#plt.plot(data[:,0], -(W_pca[0,0]/W_pca[0,1])*data[:,0], color="c")
#plt.plot(data[:,0], test2, "c")
for i in range(len(A)):
 plt.annotate(f'{i+1}', xy=(A[i]-0.1, test2[i]-0.1))

c_patch = mpatches.Patch(color='c', label='Principal components')
g_patch = mpatches.Patch(color='g', label='True components')
plt.legend(handles=[g_patch, c_patch])
plt.axis('equal')
limits = [np.minimum(np.amin(data[:,0]), np.amin(data[:,1])),
          np.maximum(np.amax(data[:,0]), np.amax(data[:,1]))]
plt.xlim(limits[0],limits[1])
plt.ylim(limits[0],limits[1])
plt.show()


x, y = np.random.multivariate_normal(mean1, np.cov(X1), 10000).T
plt.scatter(x,y)
plt.show()

#3
#тут безуспешно, мучаться дальше уже нет ни времени, ни желания
model=PCA(n_components=1)
model.fit(kernel)
W_pca=model.components_
print(model.singular_values_)
print("pca:")
print(W_pca)

trans=KernelPCA(n_components=1)
Kt=trans.fit_transform(kernel)
print(Kt)











