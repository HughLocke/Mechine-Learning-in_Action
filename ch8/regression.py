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

#基础线性回归 求出通用的模型w
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


#单样例局部加权线性回归
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

#多样例局部加权线性回归
def lwlrTest(testArr,xArr,yArr,k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#通过方差和查看错误率
def rssError(ans,Hatans):
    return ((ans - Hatans) ** 2).sum()

#岭回归
#当特征比样例多的时候,将矩阵加上一个对角线lam其他全0的矩阵
def ridgeRegres(xMat,yMat,lam = 0.2): #计算回归函数
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws
def ridgeTest(xArr,yArr,numTestPts = 30): #在一组lam上测试结果
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 标准化:所有特征减去各自的均值并除以方差,使得所有特征有相同的重要性
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0) #方差
    xMat = (xMat - xMeans) / xVar
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat
#规则化
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat
#逐步向前线性回归
def stageWise(xArr,yArr,eps = 0.01,numIt = 100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1)) #当前所有状态最优解
    wsTest = ws.copy() #尝试矩阵
    wsMax = ws.copy() #当前迭代状态下的最优解
    for i in range(numIt):
        #print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat
def main():
    xArr,yArr = loadDataSet('ex0.txt')
    paint2(xArr,yArr)


if __name__ == '__main__':
    main()