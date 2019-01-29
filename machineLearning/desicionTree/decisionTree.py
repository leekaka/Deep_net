#coding=gbk

"""

author:    PL
algorithm :DecisionTree 
Data:      20171219

"""
print(__doc__)



# DictVectorizer:��������ת��
from sklearn.feature_extraction import DictVectorizer

# csv:ԭʼ���ݷ���csv�ļ��У���packageΪpython�Դ�������Ҫ��װ
import csv

#��������Ԥ�������������������д�ַ�����
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO


#��csv�ļ��ж�ȡ���ݣ������浽allElectronicsData������
#allElectronicsData = open(r'E:\Private\Python\MachineLearning\desicionTree\data.xlsx','r')

try:
     filename = r'..\desicionTree\data3.csv'
     allElectronicsData = open(filename,'rt')
     # csv��reader�������ж�ȡ����
     reader = csv.reader(allElectronicsData)
     headers = next(reader)              ##next������ȡ��csv�ļ��ĵ�һ������

except:
    print("ʧ��")

    
print(reader)


#������list��featureListװ����ֵ��labelListװ����ǩ
featureList = []
labelList = []

#����csv�ļ���ÿһ��
for row in reader:
    #������ǩ���뵽labelList��
    labelList.append(row[len(row)-1])

    
    #�����⼸����Ŀ����Ϊ��������ֵת����һ���ֵ����ʽ���Ϳ��Ե���sk-learn�����DictVectorizer��ֱ�ӽ����������ֵת����0,1ֵ
    rowDict = {}
    
    for i in range(1,len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)

#ʵ����    
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:"+str(dummyX))
print(vec.get_feature_names())

# label��ת����ֱ����preprocessing��LabelBinarizer����
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:"+str(dummyY))
print("labelList:"+str(labelList))


#criterion��ѡ��������ڵ�ı�׼�������ǰ��ա��ء�Ϊ��׼����ID3�㷨��Ĭ�ϱ�׼��gini index����CART�㷨��
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(dummyX,dummyY)
print("clf:"+str(clf))

#����dot�ļ�
with open("decisionTree.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names = vec.get_feature_names(),out_file = f)

#���Դ��룬ȡ��1��ʵ�����ݣ���001->100����age��youth->middle_aged    
oneRowX = dummyX[0,:]
print("oneRowX:"+str(oneRowX))
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX:"+str(newRowX))
newRowX = newRowX.reshape(1,-1)
#Ԥ�����
predictedY = clf.predict(newRowX)
print("predictedY:"+str(predictedY))

