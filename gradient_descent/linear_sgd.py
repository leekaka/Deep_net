"""
随机全梯度下降方法
改进：进行到一部分的时候即更新权重

"""

import numpy as np
import time

print(__doc__)

sample = 10
num_input = 5

# 训练参数
np.random.seed(0)
normalRand = np.random.normal(0,0.1,sample)      # 10个均值为0方差为0.1 的随机数  (b)
weight = [7,99,-1,-333,0.06]                     # 1 * 5 权重
x_train = np.random.random((sample, num_input))  #x 数据（10 * 5）
y_train = np.zeros((sample,1))                   # y数据（10 * 1）


for i in range (0,len(x_train)):
	total = 0
	for j in range(0,len(x_train[i])):
		total += weight[j]*x_train[i,j]
	y_train[i] = total+ normalRand[i]


# 训练
np.random.seed(0)
weight = np.random.random(num_input+1)
rate = 0.04
batch = 3

def train(x_train,y_train):
    #计算损失
    global weight,rate
    predictY = np.zeros((len(x_train)))
    for i in range(0,len(x_train)):
        predictY[i] = np.dot(x_train[i],weight[0:num_input])+ weight[num_input]
        loss = 0
    for i in range(0,len(x_train)):
        loss += (predictY[i]-y_train[i])**2

    for i in range(0,len(weight)-1):
        grade = 0
        for j in range(0,len(x_train)):
            grade += 2*(predictY[j]-y_train[j])*x_train[j,i]
        weight[i] = weight[i] - rate*grade

    grade = 0
    for j in range(0,len(x_train)):
        grade += 2*(predictY[j]-y_train[j])
        weight[num_input] = weight[num_input] - rate*grade

    return loss

start = time.clock()
for epoch in range(0,1000):
     begin = 0
     while begin < len(x_train):
          end = begin + batch
          if end > len(x_train):
               end = len(x_train)

          loss = train(x_train[begin:end],y_train[begin:end])

          begin = end

          print("epoch: %d-loss: %f"%(epoch,loss))      #打印迭代次数和损失函数
          if loss < 0.1:
              end = time.clock()
              print("收敛时间：%s ms"%str(end-start))
              print("收敛成功%d-epoch"%epoch)
              break


print(weight)
