from sklearn import svm

X = [[2,0],[1,1],[2,3],[0,0],[3,3]]
y = [0,0,1,0,1]

clf = svm.SVC(kernel = "linear")
clf.fit(X,y)
print(clf)

print(clf.support_vectors_)
print(clf.support_)




print(clf.n_support_)
