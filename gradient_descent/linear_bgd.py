"""
普通的全梯度下降方法
"""

import numpy as np
import time

print(__doc__)

sample = 10
num_input = 5

# 加入训练数据
np.random.seed(0)
b = np.random.normal(0, 0.1, sample)      # 10个均值为0方差为0.1 的随机数  (b)
w = [7, 85, -10, -600, 20]
np.random.seed(0)                                  # 1 * 5 权重
x_train = np.random.random((sample, num_input))    # x 数据（10 * 5）
y_train = np.zeros((sample, 1))                    # y数据（10 * 1）


for i in range(0, len(x_train)):
    total = 0
    for j in range(0, len(x_train[i])):
        total += w[j] * x_train[i, j]
    y_train[i] = total + b[i]

# 训练
np.random.seed(0)
weight = np.random.random(num_input+1)

rate = 0.07
start = time.clock()
for epoch in range(0, 5000):                     # 计算loss
    predictY = np.zeros((len(x_train)))
    for i in range(0, len(x_train)):
        predictY[i] = np.dot(x_train[i], weight[0:num_input]) + weight[num_input]
    loss = 0
    for i in range(0, len(x_train)):
        loss += (predictY[i]-y_train[i]) ** 2

    # print("epoch: %d-loss: %f" % (epoch, loss))    # 打印迭代次数和损失函数
    # 计算梯度并更新
    for i in range(0, len(weight)-1):                               # 权重w
        grade = 0
        for j in range(0, len(x_train)):
            grade += (predictY[j]-y_train[j]) * x_train[j, i]
        weight[i] = weight[i] - rate*grade
    grade = 0
    for j in range(0, len(x_train)):                                 # 偏差b
        grade += (predictY[j]-y_train[j])
    weight[num_input] = weight[num_input] - rate*grade


end = time.clock()
print("迭代时间:%s ms" % str(end-start))
print("epoch: %d-loss: %f" % (epoch, loss))
print(weight)
