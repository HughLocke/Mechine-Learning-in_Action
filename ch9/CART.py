from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for Line in fr.readlines():
        curLine = Line.strip().split('\t')
        fltLine = []
        for i in curLine:
            fltLine.append(float(i))
        dataMat.append(fltLine)
    return dataMat

#二元切分数据集
#将数据集上所有在feature特征的数据以value为界限划分为mat0和mat1
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#选出划分特征的下标和划分的值
def chooseBestSplit(dataSet,leafType = regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #这个特征下全是相同的值
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf;bestIndex = 0;bestValue = 0
    for featIndex in range(n - 1): #遍历所有属性
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]): #遍历这个属性下所有可能的值
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal) #尝试划分的两个矩阵
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0]<tolN): continue #划分之后其中一个数据集过小,则不考虑
            newS = errType(mat0) + errType(mat1) #重新计算误差
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS: #降低误差的范围小到一定程度,就不管了
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #无法划分出满足另一个数据集条件的情况,也不管了
        return None,leafType(dataSet)
    return bestIndex,bestValue
#数据集.建立叶节点的函数,误差计算函数,包含树构建所需其他参数的元祖
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops = (1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val #不需要划分,直接返回这个叶子节点的答案
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree
#后剪枝
#通过判断儿子是不是字典来判断是树还是值
def isTree(obj):
    return (type(obj).__name__ == 'dict')
#合并两树
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree[tree['left']]: tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0
def prune(tree,testData):
    if shape(testData)[0] == 0.0: return getMean(tree)
    if (isTree(tree['right']) or isTree[tree['left']]):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNomerge = sum(power(lSet[:,-1] - tree['left'],2))

def main():
    dataset = loadDataSet('ex00.txt')
    dataMat = mat(dataset)
    root = createTree(dataMat,ops = (0,1))
    print(root)

if __name__ == '__main__':
    main()