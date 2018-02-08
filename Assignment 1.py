
# coding: utf-8

# In[216]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


# In[300]:

# Generating Data for training: 


# In[360]:

x=10*np.transpose(np.array([np.random.randn(1000),np.random.randn(1000)]))
w1=np.random.rand()
w2=2*np.random.rand()
y=w1*x[:,0]+w2*x[:,1]+2*np.random.randn(1000)

plt.plot(x[:,0],y,'ro')
plt.xlabel('$x_1$')
plt.ylabel('$y$')
plt.show()

print [w1,w2]


# In[361]:

class LinearRegressionClass():
    
    def __init__(self, learning_rate=0.01, stopping_criterion=0.01, max_iterations=1000,max_epochs=100,discounting=0.9,momentum_coef=0.1):
        self.learning_rate = learning_rate
        self.stopping_criterion = stopping_criterion
        self.max_iterations = max_iterations
        self.max_epochs = max_epochs
        self.discounting = discounting
        self.momentum_coef=momentum_coef
        self.w = None
    
    def SGD_fit(self,X,Y):
        dim=X.shape
        n=dim[0]
        m=dim[1]

        self.w=np.random.randn(m)
        loss=np.zeros(self.max_epochs)
        for i in range(0,self.max_epochs):
            for j in range(0,n):
                self.w-=self.learning_rate*X[j,:]*(np.dot(self.w,X[j,:]) - Y[j])/n
                loss[i]+=(np.dot(self.w,X[j,:]) - Y[j])**2
        plt.plot(loss[:self.max_epochs])
        plt.ylabel('Loss')
        plt.xlabel('# epochs')
        plt.show()
        return self.w
    
    def SGDMomentum_fit(self,X,Y):
        dim=X.shape
        n=dim[0]
        m=dim[1]
        
        dw=np.zeros(m)
        self.w=np.random.randn(m)
        loss=np.zeros(self.max_epochs)
        
        for i in range(0,self.max_epochs):
            for k in range(0,n):
                self.w-=self.learning_rate*X[k,:]*(np.dot(self.w,X[k,:]) - Y[k])/n
                self.w+=self.momentum_coef*dw
                dw=X[k,:]*(np.dot(self.w,X[k,:]) - Y[k])/n
                loss[i]+=(np.dot(self.w,X[k,:]) - Y[k])**2
        plt.plot(loss[:self.max_epochs])
        plt.ylabel('Loss')
        plt.xlabel('# epochs')
        plt.show()
        return self.w
    
    def SGDAdaGrad_fit(self,X,Y):
        dim=X.shape
        n=dim[0]
        m=dim[1]
    
        cumm_grad=np.zeros(m)+0.000001
        self.w=np.random.randn(m)
        loss=np.zeros(self.max_epochs)
        for i in range(0,self.max_epochs):
            for j in range(0,n):
                self.w-=X[j,:]/np.sqrt(cumm_grad)*self.learning_rate*(np.dot(self.w,X[j,:]) - Y[j])/n
                cumm_grad+=np.square(X[j,:]*(np.dot(self.w,X[j,:]) - Y[j])/n)
                loss[i]+=(np.dot(self.w,X[j,:]) - Y[j])**2
        plt.plot(loss[:self.max_epochs])
        plt.ylabel('Loss')
        plt.xlabel('# epochs')
        plt.show()
        return self.w
    
    def SGDRMSProp_fit(self,X,Y):
        dim=X.shape
        n=dim[0]
        m=dim[1]

        self.w=np.random.randn(m)
        v=0
        loss=np.zeros(self.max_epochs)
        for i in range(0,self.max_epochs):
            for j in range(0,n):
                v=self.discounting*v+(1-self.discounting)*np.sum((X[j,:]*(np.dot(self.w,X[j,:]) - Y[j])/n)**2)
                self.w-=self.learning_rate*X[j,:]*(np.dot(self.w,X[j,:]) - Y[j])/(n*np.sqrt(v))
                loss[i]+=(np.dot(self.w,X[j,:]) - Y[j])**2
        plt.plot(loss[:self.max_epochs])
        plt.ylabel('Loss')
        plt.xlabel('# epochs')
        plt.show()
        return self.w
    
    def BGD_fit(self,X,Y):
        dim=X.shape
        n=dim[0]
        m=dim[1]

        w=np.random.randn(m)
        grad=2*self.stopping_criterion*np.ones(m)
        loss=np.zeros(self.max_iterations)

        count=0
        while(np.linalg.norm(grad)>self.stopping_criterion and count<self.max_iterations):
            for k in range(0,n):
                loss[count]+=(Y[k]-np.dot(X[k,:],w))**2/n
            for j in range(0,m):
                grad[j]=0
                for k in range(0,n):
                    grad[j]+=(Y[k]-np.dot(X[k,:],w))*X[k,j]/n

                w[j]+=self.learning_rate*grad[j]
            count+=1
        self.w=w
        plt.plot(loss[:count])
        plt.ylabel('Loss')
        plt.xlabel('# iteration')
        plt.show()
        return self.w
    
    def predict(self,X):
        return np.dot(X,self.w)
    
    def score(self,X,Y):
        return 1-np.var(Y-np.dot(X,self.w))/np.var(Y)


# In[362]:

linear_model=LinearRegressionClass()


# In[ ]:

# Batch gradient descent:


# In[364]:

linear_model.BGD_fit(x,y)


# In[365]:

linear_model.score(x,y)


# In[366]:

# Plain Stochastic gradient descent:


# In[367]:

linear_model.SGD_fit(x,y)


# In[368]:

linear_model.score(x,y)


# In[369]:

# Stochastic gradient descent with Momentum:


# In[370]:

linear_model.learning_rate=0.1
linear_model.SGDMomentum_fit(x,y)


# In[371]:

linear_model.score(x,y)


# In[372]:

# Stochastic gradient descent with Adaptive gradient:


# In[376]:

linear_model.max_epochs=1000
linear_model.learning_rate=0.1
linear_model.SGDAdaGrad_fit(x,y)


# In[377]:

linear_model.score(x,y)


# In[297]:

# Stochastic gradient descent with RMS Propogation:


# In[378]:

linear_model.SGDRMSProp_fit(x,y)


# In[379]:

linear_model.score(x,y)



