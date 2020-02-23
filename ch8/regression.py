from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  #计算行列式
        print("This mat is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


def paint(xArr,yArr):
    import matplotlib.pyplot as plt
    ws = standRegres(xArr,yArr)
    #print(ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()
    yHat = xMat * ws
    print(corrcoef(yHat.T,yMat))

def lwlr(testPoint,xArr,yArr,k=1.0): #k控制距离变长权重衰减的速度
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]  #样本数量
    weights = mat(eye((m)))
    for j in range(m): #计算每个样本对这个数据的权重，距离越远的权重以指数级下降
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) #高斯核
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This Matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def paint2(xArr,yArr):

    import matplotlib.pyplot as plt
    ws = standRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0],s = 2,c = 'red')
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = lwlrTest(xCopy,xArr,yArr,0.002)
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

def main():
    xArr,yArr = loadDataSet('ex0.txt')
    paint2(xArr,yArr)


if __name__ == '__main__':
    main()