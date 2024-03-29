from model import *
from dense import *
from activation import *
from loss import *
from optimizer import *
import numpy as np

# xor

# 加入训练数据
x_train = np.array([[0,1],[1,0],[0,0],[1,1]])
y_train = np.array([[0],[0],[0],[1]])

# 训练
model = Model()
model.add(Dense(3,input_shape=(1,2)))
model.add(Activation("sigmoid"))

model.add(Dense(5))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

optimizer = Optimizer(rate=0.1,momentum=0.9)
loss = Loss("mse")
model.compile(optimizer=optimizer,loss=loss)
model.fit(x_train,y_train,epochs=1000)

for i in range(0,len(x_train)):
	print("data :%d,%d,%f"%(x_train[i,0],x_train[i,1],model.predict(x_train[i])))
print(model.get_weight())
