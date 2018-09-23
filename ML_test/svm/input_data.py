import numpy as np
import matplotlib.pyplot as pl
from sklearn import svm


np.random.seed(0)

X = np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2) + [2,2]]

print(X)

y = np.r_[1,1]
print(y)
