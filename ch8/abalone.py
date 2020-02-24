from regression import *

#预测鲍鱼年龄
def rssError(ans,Hatans):
    return ((ans - Hatans) ** 2).sum()


def main():
    dataset,ans = loadDataSet('abalone.txt')
    #局部加权线性回归
    ansHat01 = lwlrTest(dataset[100:199],dataset[0:99],ans[0:99],0.1)
    ansHat1 = lwlrTest(dataset[100:199],dataset[0:99],ans[0:99],1)
    ansHat10 = lwlrTest(dataset[100:199],dataset[0:99],ans[0:99],10)
    print(rssError(ans[100:199], ansHat01.T))
    print(rssError(ans[100:199], ansHat1.T))
    print(rssError(ans[100:199], ansHat10.T))
    #基础线性回归
    ws = standRegres(dataset[0:99],ans[0:99])
    ansHat = mat(dataset[100:199]) * ws
    print(ansHat.T)
    print(ansHat.T.A)
    print(rssError(ans[100:199],ansHat.T.A))
if __name__ == '__main__':
    main()
