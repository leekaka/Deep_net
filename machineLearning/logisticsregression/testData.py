import numpy as np
import random

def gradientDescent(x,y,theta,alpha,m,numIterations):    #梯度下降法
    xTrans = x.transpose()    #转秩

    
    for i in range(0,numIterations):     #

        hypothesis = np.dot(x,theta)

        loss = hypothesis - y

        cost = np.sum(loss ** 2)/(2 * m)

        print("Iteration %d / Cost: %f" %(i,cost))

        gradient = np.dot(xTrans,loss)/m

        theta = theta - alpha*gradient


        
    return theta




def genData(numPoints,bias,variance):
    x = np.zeros(shape=(numPoints,2))   #初始化[0,0]  numPoints行，2列
    y = np.zeros(shape=numPoints)       #初始化 y 一行 numPoints 列    

    for i in range(0,numPoints):
        x[i][0] = 1                     #第i行的第一个数是 0 ，第二个数是 i
        x[i][1] = i

        y[i] = (i+bias)+random.uniform(0,1)*variance  #均匀分布的0，1的值  乘以10  ，单增的数值加了一个10以内的偏差
        #y[i] = (i+bias)
    return x,y  #创造了数据集，两个x,第一个x不变，第二个x单增，y单增

x,y = genData(100,25,10)

#print("x:")
#print(x)
#print("y:")
print(y)
m,n = np.shape(x)  #m=100,n=2
n_y = np.shape(y)   #n_y=(100,)

numIterations = 100000
alpha = 0.000005
theta = np.ones(n)   #theta = [1. 1.] 初始化

theta = gradientDescent(x,y,theta,alpha,m,numIterations)

