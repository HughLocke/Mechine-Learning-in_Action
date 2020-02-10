from math import log
import operator
def CreateDataSet(): #用于测试代码的数据集
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['nosurfacing','flippers','fish']
    return dataset,labels

def Ent(dataSet):  #求信息熵
    n = len(dataSet)
    labelCounts = {}
    for Sample in dataSet:
        ans = Sample[-1]
        if ans not in labelCounts.keys():
            labelCounts[ans] = 0
        labelCounts[ans] += 1
    Entans = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / n
        Entans -= prob * log(prob,2)
    return Entans

def splitDataSet(dataSet,axis,value):   #划分数据集(返回下标属性axis为value的数据集)
    retDataSet = []
    for Sample in dataSet:
        if Sample[axis] == value:
            tmpSample = Sample[:axis]
            tmpSample.extend(Sample[axis + 1:])
            retDataSet.append(tmpSample)
    return retDataSet

def chooseBestFeatureTosplit(dataSet): #返回当前节点最优划分属性的下标
    numFeatures = len(dataSet[0]) - 1     #可能的分支
    baseEnt = Ent(dataSet)  #当前数据集的信息熵
    bestGain = 0.0 #目前最多的信息增益
    bestFeature = -1   #目前最好的分支
    for i in range(numFeatures): #遍历所有属性
        featList = [example[i] for example in dataSet] #选出该属性所有的值作为列表
        uniqueVals = set(featList)   #利用set去重
        newEnt = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEnt += prob * Ent(subDataSet)
        Gain = baseEnt - newEnt
        if bestGain < Gain:
            bestGain = Gain
            bestFeature = i
    return bestFeature

def majorityCnt(classList): #当前子集中最多的元素
    classCount = {}
    for i in classList:
        if i not in classCount:
            classCount[i] = 0
        classCount[i] += 1
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]

def createTree(dataSet,labels): #创建树,返回当前节点,如果是叶子节点，则直接返回结果
    anslist = [example[-1] for example in dataSet] #结果集
    if anslist.count(anslist[0]) == len(anslist):  #子集中全是相同结果
        return anslist[0]
    if len(dataSet[0]) == 1:    #当前只有一个属性,则返回占多数的结果
        return majorityCnt(anslist)
    bestFeatindex = chooseBestFeatureTosplit(dataSet)  #选出接下来的划分属性的下标
    #print(bestFeatindex)
    bestFeatlabel = labels[bestFeatindex] #划分属性标签
   # print(bestFeatlabel)
    node = {bestFeatlabel:{}}
    del(labels[bestFeatindex]) #接下来就不需要划分这个属性了,删除
    featValues = [example[bestFeatindex] for example in dataSet] #列出数据集中所有划分属性的元素
    uniqueVals = set(featValues) #去重
    for value in uniqueVals:
        subLabels = labels[:]
        tmplabels = labels.copy()
        node[bestFeatlabel][value] = createTree(splitDataSet(dataSet,bestFeatindex,value),tmplabels)
    return node

def zerokey(myTree):
    for key in myTree:
        return key

def classify(inputTree,featLabels,testVec): #分类函数
    firstStr = zerokey(inputTree)
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                ans = classify(secondDict[key],featLabels,testVec)
            else:
                ans = secondDict[key]
    return ans

def storeTree(inputTree,filename): #通过文件存储决策树
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def main():
    Dataset, label = CreateDataSet()
    # print(Ent(dataset))
    # print(splitDataSet(dataset,1,1))
    # print(chooseBestFeatureTosplit(dataset))
    tmp = label.copy()
    tree = createTree(Dataset, tmp)
    print(classify(tree,label,[1,1]))

if __name__ == '__main__':
    main()