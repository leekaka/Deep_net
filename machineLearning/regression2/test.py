'''
genfromtxt('waveform.txt',delimiter=',',skip_header=18)
多元线性回归

'''
from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

print(__doc__)
dataPath = r"E:\Private\Python\MachineLearning\regression2\Delivery.txt"
deliveryData = genfromtxt(dataPath,delimiter='\t')   #以\t来分割数据


#print(deliveryData)

X = deliveryData[:, :-1]   #排除倒数第几列
Y = deliveryData[:, -1]   #最后一列

#z = deliveryData[:,:2]   #顺数第几列
#z = deliveryData[:2,:]    #顺数第几行
#z = deliveryData[:-2,:]   #排除倒数第几行
#z = deliveryData[1:-2,:2]

print ("X:")
print (X)
print ("Y: ")
print (Y)


regr = linear_model.LinearRegression()
regr.fit(X, Y)

print ("coefficients")
print (regr.coef_)
print ("intercept: ")
print (regr.intercept_)

xPred = [102, 6]

xPred = np.array(xPred).reshape(1,-1)  #2D数据

yPred = regr.predict(xPred)


print ("predicted y: ")
print (yPred)
