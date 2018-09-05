from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

import matplotlib.pyplot as plt

def main():
	x_train = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
	y_train = np.array([[0],[1],[0],[0]])

	model = Sequential()

	model.add(Dense(units = 2,input_dim = 2))
	model.add(Activation("relu"))

	model.add(Dense(units = 1))
	model.add(Activation("sigmoid"))

	model.compile(loss='binary_crossentropy',optimizer = 'sgd',metrics=['accuracy'])

	hist = model.fit(x_train,y_train,epochs = 1000)

	plt.scatter(range(len(hist.history['loss'])), hist.history['loss'])

	#测试数据
	loss_and_metrics = model.evaluate(x_train, y_train)
	print(loss_and_metrics)

	plt.show()

	
if __name__ == '__main__':\
		main()
