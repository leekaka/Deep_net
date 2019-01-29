from sklearn import neighbors
from sklearn import datasets

# get knn algorithm
knn = neighbors.KNeighborsClassifier()
# get iris datasets
iris = datasets.load_iris()

#print(iris)

# use knn build the model
knn.fit(iris.data,iris.target)

# use the model to predict
predictedLabel = knn.predict([[0.1,0.2,0.3,0.4]])
print(predictedLabel)
