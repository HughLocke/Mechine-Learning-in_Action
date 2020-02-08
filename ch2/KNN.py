from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#dataSet:训练样本集合,labels:训练样本的分类,inX:用于分类的输入向量(一个测试数据)
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]   #样例的大小
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #将测试数据扩充为一个列表,每个元素diffMat_i = inX_i - dataSet_i,相当于将测试数据分别对每个训练数据作差
    sqDiffMat = diffMat * 2
    sqDiffMat = sqDiffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    # 分别求出测试数据对每个训练数据的欧式距离

    sortedDistIndicies = distances.argsort() #返回数组从小到大的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
    key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def filetomatrix(filename): #将文本格式的数据处理成numpy的解析程序
    fr = open(filename)
    Array_Of_Lines = fr.readlines()
    Number_Of_Lines = len(Array_Of_Lines)
    returnMat = zeros((Number_Of_Lines,3))#创建一个行数为Lines,3列的全0矩阵
    classLabelVector = []
    index = 0
    x = -1
    for line in Array_Of_Lines:
        line = line.strip()
        ListFromLine = line.split('\t')
        returnMat[index,:] = ListFromLine[0:3]
        classLabelVector.append(int(ListFromLine[-1]))
        x += 1
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):   #归一化
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals  #最大最小值的差
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat,datingLabels = filetomatrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    Testnum = int(m*hoRatio)
    errorCount = 0.0
    for i in range(Testnum):
        ans = classify0(normMat[i,:],normMat[Testnum:m,],
                                     datingLabels[Testnum:m],3)
        print("the classifier came back with %d,the real answer is: %d"
              %(ans,datingLabels[i]))
        if ans != datingLabels[i] :
            errorCount += 1
    print("the total error rate is: %f"  % (errorCount/float(Testnum)))

if __name__ == '__main__':
    '''
    datingDataMat,datingLabels = filetomatrix("datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    datingLabels2 = ['r' if i == 1 else i for i in datingLabels]
    datingLabels2 = ['g' if i == 2 else i for i in datingLabels2]
    datingLabels = ['b' if i == 3 else i for i in datingLabels2]
    ax.scatter(datingDataMat[:, 0], datingDataMat[:,1],
               15,c = datingLabels)
    plt.show()
    '''
    datingClassTest()
