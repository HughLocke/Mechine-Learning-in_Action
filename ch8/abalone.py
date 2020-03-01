from regression import *

#预测鲍鱼年龄

def way1(dataset,ans):
    ansHat01 = lwlrTest(dataset[100:199], dataset[0:99], ans[0:99], 0.1)
    ansHat1 = lwlrTest(dataset[100:199], dataset[0:99], ans[0:99], 1)
    ansHat10 = lwlrTest(dataset[100:199], dataset[0:99], ans[0:99], 10)
    print(rssError(ans[100:199], ansHat01.T))
    print(rssError(ans[100:199], ansHat1.T))
    print(rssError(ans[100:199], ansHat10.T))

def way2(dataset,ans):
    ws = standRegres(dataset[0:99], ans[0:99])
    print(ws)
    ansHat = mat(dataset[100:199]) * ws
    #print(rssError(ans[100:199], ansHat.T.A))

def way3(dataset,ans):
    ridgeWeights = ridgeTest(dataset, ans)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def way4(dataset,ans):
    ws = stageWise(dataset, ans, 0.001, 5000)[4500]
    ws = mat(ws).T
    print(ws)
    ansHat = mat(dataset[100:199]) * ws
    #print(rssError(ans[100:199],ansHat.T.A))

def main():
    dataset,ans = loadDataSet('abalone.txt')
    #局部加权线性回归
    way1(dataset,ans)
    #基础线性回归
    way2(dataset,ans)
    #岭回归
    way3(dataset,ans)
    #逐步线性回归
    way4(dataset,ans)
if __name__ == '__main__':
    main()
