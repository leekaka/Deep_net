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
b = np.random.normal(0, 0.1, sample)             # 10个均值为0方差为0.1 的随机数  (b)
w = [7, 85, -10, -600, 20]                 # 1 * 5 权重
x_train = np.random.random((sample, num_input))  # x 数据（10 * 5）
y_train = np.zeros((sample, 1))                  # y数据（10 * 1）

for i in range(0, len(x_train)):
    total = 0
    for j in range(0, len(x_train[i])):
        total += w[j]*x_train[i, j]
    y_train[i] = total + b[i]

# 训练w
np.random.seed(0)
weight = np.random.random(num_input+1)
rate = 0.07
batch = 4

def train(x, y):
    # 计算损失
    global weight, rate
    predict_y = np.zeros((len(x)))
    for i_ in range(0, len(x)):
        predict_y[i_] = np.dot(x[i_], weight[0:num_input]) + weight[num_input]
    ls = 0
    for i_ in range(0, len(x)):
        ls += (predict_y[i_] - y[i_])**2
        
    for i_ in range(0, len(weight)-1):
        grade = 0
        for j_ in range(0, len(x)):
            grade += 2*(predict_y[j_] - y[j_])*x[j_, i_]
        weight[i_] = weight[i_] - rate*grade
    grade = 0
    for j_ in range(0, len(x)):
        grade += 2*(predict_y[j_] - y[j_])
        weight[num_input] = weight[num_input] - rate*grade
    return ls


s = time.time()
for epoch in range(0, 5000):
    begin = 0
    while begin < len(x_train):
        end = begin + batch
        if end > len(x_train):
            end = len(x_train)
        loss = train(x_train[begin:end], y_train[begin:end])
        begin = end
e = time.time()
print("迭代时间：%s ms" % str(e - s))
print("epoch: %d-loss: %f" % (epoch, loss))
print(weight)
