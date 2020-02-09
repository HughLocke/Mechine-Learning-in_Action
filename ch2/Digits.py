import numpy as np
import os
import KNN

def imagetovector(filename):
    returnVector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,i*32+j] = int(lineStr[j])
    return returnVector

def handwritingClassTest():
    trainingLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        trainingLabels.append(classNumstr)
        trainingMat[i,:] = imagetovector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    n = len(testFileList)
    for i in range(n):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        ans = int(fileStr.split('_')[0])
        testvector = imagetovector('testDigits/%s' % fileNameStr)
        KNNans = KNN.classify0(testvector,trainingMat,trainingLabels,3)
        #print(f"true ans = {ans},KNNans = {KNNans}")
        if ans != KNNans:
            errorCount += 1
    print(f"number of errror = {errorCount}")
    print(f"error rate is {errorCount/float(n)}")

if __name__ == '__main__':
    handwritingClassTest()