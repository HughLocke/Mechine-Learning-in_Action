from numpy import *
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m): #寻找[0,m)之间不等于i的一个随机数
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L): #调整大于H或者小于L的alpha值
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataset,label,C,toler,maxIter):
    dataMat = mat(dataset)
    labelMat = mat(label).transpose()
    b = 0
    m,n = shape(dataMat) #数据集的行列数
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter): #迭代次数
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMat*dataMat[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMat*dataMat[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L==H:
                    print("L==H");
                    continue
                eta = 2.0 * dataMat[i,:]*dataMat[j,:].T-dataMat[i,:]*dataMat[i,:].T - \
                    dataMat[j,:] * dataMat[j,:].T


def main():
    dataSet,label = loadDataSet('testSet.txt')
    smoSimple(dataSet,label)


if __name__ == '__main__':
    main()