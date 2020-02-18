import numpy as np

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  #转换矩阵后转置
    print(labelMat)
    print(np.mat(classLabels))
    m,n = np.shape(dataMatrix)  #m:数据集个数,n:属性个数
    alpha = 0.001       #梯度变化系数
    maxCycles = 500      #梯度变化次数
    weights = np.ones((n,1))  #初始化回归系数为1列n行的全1矩阵
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h   #计算预测类别和真实类别的差别
     #   print(error)
        weights = weights + alpha * dataMatrix.transpose() * error
        #print(weights)
    return weights

def main():
    dataSet,labelSet = loadDataSet()
    gradAscent(dataSet,labelSet)
if __name__ == '__main__':
    main()