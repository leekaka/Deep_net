from e import NeuralNetwork
import numpy as np

nn = NeuralNetwork([3,2,1],'logistic')   #三层，第一层有两个。第二层两个，第三层一个



X = np.array([[0,0,0],[0,1,1],[1,0,0],[1,1,1]])
y = np.array([0,1,1,0])


nn.fit(X,y)
#for i in [[0,0],[0,1],[1,0],[1,1]]:
    #print(i,nn.predict(i))
    #pass

