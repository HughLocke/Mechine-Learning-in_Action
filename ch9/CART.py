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

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType = regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf;bestIndex = 0;bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0]<tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
        if (S - bestS) < tolS:
            return None,leafType(dataSet)
        mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
        if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
            return None,leafType(dataSet)
        return bestIndex,bestValue
#数据集.建立叶节点的函数,误差计算函数,包含树构建所需其他参数的元祖
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops = (1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

def main():
    dataset = loadDataSet('ex00.txt')
    dataMat = mat(dataset)
    root = createTree(dataMat)
    print(root)

if __name__ == '__main__':
    main()