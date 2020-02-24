from regression import *
from numpy import *
import random as rd
def way1(dataset,ans):
    n, m = shape(dataset)
    dataset1 = mat(ones((n, m + 1)))
    dataset1[:, 1:5] = mat(dataset)  # 加入一列全为1的属性,目的是代表偏移常数
    ws = standRegres(dataset1, ans)
    print(mat(ws).T)

def crossValidation(dataset,ans,numVal = 10): #交叉测试岭回归
    n = len(ans)
    indexList = [i for i in range(n)]
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        traindata = []; trainans = []
        testdata = []; testans = []
        rd.shuffle(indexList) #打乱下标
        for j in range(n):
            if j < n * 0.9: #取出前90%作为训练集
                traindata.append(dataset[indexList[j]])
                trainans.append(ans[indexList[j]])
            else:      #剩下的10%数据作为训练集
                testdata.append(dataset[indexList[j]])
                testans.append(ans[indexList[j]])
        wMat = ridgeTest(traindata,trainans) #一组岭回归返回的结果
        for k in range(30): #对每个lam返回的结果进行误差分析
            #用训练集的参数将测试集标准化
            matTestdata = mat(testdata); matTraindata = mat(traindata)
            meanTrain = mean(matTraindata,0)
            varTrain = var(matTraindata,0)
            matTestdata = (matTestdata - meanTrain) / varTrain
            #利用标准化后的测试集乘以模型,然后加上训练集答案的平均值以返回预测答案
            yEst = matTestdata * mat(wMat[k,:]).T + mean(trainans)
            errorMat[i,k] = rssError(yEst.T.A,array(testans))
    #所有lam下的10组测试平均误差和
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors)) #找出最小误差
    bestWeights = wMat[nonzero(meanErrors == minMean)] #找出最小误差的lam对应的模型
    dataMat = mat(dataset)
    ansMat = mat(ans).T
    meanX = mean(dataMat,0)
    varX = var(dataMat,0)
    unReg = bestWeights/varX
    c = -1*sum(multiply(meanX,unReg)) + mean(ansMat) #计算常数项
    print(meanErrors)
    print(dataset[5] * unReg.T + c)
    print(ans[5])
    print("the best model from Ridge Regression is :\n",unReg)
    print("with constant term: ",c)
def main():
    dataset,ans = loadDataSet('LEGO.txt')
    #way1(dataset,ans) #基础线性回归
    crossValidation(dataset,ans,10)
if __name__ == '__main__':
    main()