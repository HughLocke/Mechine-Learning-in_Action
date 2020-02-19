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

def sigmoid(inX): #大于0.5被分如入1,小于0.5被分入0的函数
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  #转换矩阵后转置
    m,n = np.shape(dataMatrix)  #m:数据集个数,n:属性个数

    alpha = 0.001       #梯度变化系数
    maxCycles = 500      #梯度变化次数
    weights = np.ones((n,1))  #初始化回归系数为1列n行的全1矩阵
   # print(n)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h   #计算预测类别和真实类别的差别
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights): #画出回归最佳拟合直线的函数
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0, 3.0, 0.1) #起始,终点,步长
    y = (-weights[0]-weights[1]*x)/weights[2] #每个点的y坐标
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix,classLabels): #随机梯度上升算法
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#进行numIter次迭代,每次按照随机顺序将所有样本进行梯度下降
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = [i for i in range(m)]
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex))) #随机选择样本进行梯度下降
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def main():
    dataSet,labelSet = loadDataSet()
    weights = stocGradAscent1(np.array(dataSet),labelSet)
    plotBestFit(weights)

if __name__ == '__main__':
    main()