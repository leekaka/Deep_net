'''
普通的全梯度下降方法
'''
import numpy as np
import math 
import time

print(__doc__)

sample = 10
num_input = 5

#加入训练数据
np.random.seed(0)
normalRand = np.random.normal(0,0.1,sample)      # 10个均值为0方差为0.1 的随机数  (b)
weight = [5,100,-5,-400,0.02]  
np.random.seed(0)                  # 1 * 5 权重
x_train = np.random.random((sample, num_input))  #x 数据（10 * 5）
y_train = np.zeros((sample,1))                   # y数据（10 * 1）


for i in range (0,len(x_train)):
	total = 0
	for j in range(0,len(x_train[i])):
		total += weight[j]*x_train[i,j]
	y_train[i] = total+ normalRand[i]

print(y_train)
	

# 训练
np.random.seed(0)
weight = np.random.random(num_input+1)
np.random.seed(0)
recordGrade  = np.random.random(num_input+1)

discount = 0.9
rate = 0.02


start = time.clock()
for epoch in range(0,500):
	#预先走一步，计算下一个的权重w
	oldweight = np.copy(weight)
	for i in range(0,len(weight)):
		weight[i] = weight[i] - rate * discount *  recordGrade[i]
	
	

	# 计算loss
	predictY = np.zeros((len(x_train)))	
	for i in range(0,len(x_train)):
		predictY[i] = np.dot(x_train[i],weight[0:num_input])+ weight[num_input]
		
	loss = 0

	
	for i in range(0,len(x_train)):
		loss += (predictY[i]-y_train[i])**2		
	print("epoch: %d-loss: %f"%(epoch,loss))                          #打印迭代次数和损失函数
	
	if loss < 0.1:
		end = time.clock()
		print("收敛时间：%s ms"%str(end-start))
		print("收敛成功%d-epoch"%epoch)
		break

	# 计算梯度并更新
	weight = oldweight
	
	for i in range(0,len(weight)-1):                                  #权重w
		grade = 0
		for j in range(0,len(x_train)):
			grade += 2*(predictY[j]-y_train[j])*x_train[j,i]
		#计算梯度用下一个权重，计算权重用原来的
		recordGrade[i] = recordGrade[i]*discount + grade
		weight[i] = weight[i] - rate*recordGrade[i]

	grade = 0
	for j in range(0,len(x_train)):                                   #偏差b
             
		grade += 2*(predictY[j]-y_train[j])
	recordGrade[num_input]	 = recordGrade[num_input]*discount + grade
	weight[num_input] = weight[num_input] - rate*recordGrade[num_input]


#预测
#predict
y_predict = np.zeros(len(x_train))
for i in range (0,len(x_train)):
	total = 0
	for j in range(0,len(x_train[i])):
		total += weight[j]*x_train[i,j]
	y_predict[i] = total+ normalRand[i]
	
for i in range(0,len(y_train)):
        print(y_predict[i] - y_train[i])










