import csv
import random
import math
import operator


# 读取数据并且将数据分成了训练集和测试集

def loadDataset(filename,split,trainingSet=[],testSet=[]):
    
    with open(filename,'rt') as csvfile:
        lines = csv.reader(csvfile)  #读取数的每一行     
        dataSet = list(lines)   #将数据变成了列
      

        for x in range(len(dataSet)):  #0，149 (150个)
            for y in range(4):
                dataSet[x][y] =float(dataSet[x][y])   #数据类型需要是float才能进行数据处理 
       
            if random.random() < split:
                trainingSet.append(dataSet[x])
            else:
                testSet.append(dataSet[x])

        
def euclideanDistance(instance1,instance2,length):
    distance = 0;
    for x in range(length):
        distance +=pow(instance1[x] - instance2[x],2)
    return math.sqrt(distance)


def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance)-1       #testInstance是一个实例数据

    for x in range(len(trainingSet)):  #trainingSet的每组数据(训练的所有数据和测试的数据做距离计算)

        dist = euclideanDistance(testInstance,trainingSet[x],length)  
        distances.append((trainingSet[x],dist))  #把dist加进去了(([a, b, c, d, 'Iris-setosa'], dist))
    distances.sort(key = operator.itemgetter(1)) #将测试集所有数据和Instance的距离求出来排序
    #print(distances) 
    
    neighbors = []

    
    for x in range(k):
        #print(distances[x][0])
        #print(distances[x][1])
        neighbors.append(distances[x][0])  #只取k个并且将列在后面的dist抛弃（取第一个数据）
    
    return neighbors
    

def getResponse(neighbors):
    classVotes = {}   #字典

    for x in range(len(neighbors)):
        response = neighbors[x][-1]    #最后一个，即目标
        
        #print(response)
        
        if response in classVotes:   #如果没有出现，则为1，出现过了，每出现一次加1
            classVotes[response] += 1
        else:
            classVotes[response] = 1
            
        sortedVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse = True)
        #按字典里的第二个参数进行排序，即是出现的次数
        
    #print(sortedVotes[0][0])
    #print(classVotes.items())
    #print(sortedVotes)
    
    return sortedVotes[0][0]  #给出排列在第一的目标

def getAccuracy(testSet,predictions):
    correct = 0
    wrong = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
        else:
            wrong  += 1
    return (correct/float(len(testSet)) *100.0)



def main():
    trainingSet = []
    testSet = []
    split = 0.66667

    
    loadDataset(r'..\knn\iris.txt',split,trainingSet,testSet)
    
    print('Train set:' +str(len(trainingSet)))
    print('Test set:' +repr(len(testSet)))

    predictions = []
    
    k = 10                         #在排序的顺序里只选三个用来投票

    t = 0                          #计算错误的个数
    
    for x in range(len(testSet)):  # 对测试集里的每一组数据进行判断     
        neighbors = getNeighbors(trainingSet,testSet[x],k)
        
        result = getResponse(neighbors)
        
        predictions.append(result)
        
        print('>predicted=' +repr(result) +',actual = '+repr(testSet[x][-1]))
        
        if testSet[x][-1] != predictions[x]:   #如果测试和预测的不对
            t += 1
            print('wrong '+'>predicted=' +repr(result) +',actual = '+repr(testSet[x][-1]))


            
    accuracy =getAccuracy(testSet,predictions) 
    print('Accuracy:' +repr(accuracy) +'%')

    print('WrongNum ' +str(t))


    
main()

          

