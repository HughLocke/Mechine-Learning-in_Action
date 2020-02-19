import numpy as np
import logistic as lg

def classifyVector(inX,weights):
    prob = lg.sigmoid(sum(inX*weights))
    #prob = 0
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open('horseColicTest.txt')
    TrainSet = []; TrainLabel = []
    TestSet = []; TestLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): lineArr.append(float(currLine[i]))
        TrainSet.append(lineArr)
        TrainLabel.append(float(currLine[21]))
    trainWeights = lg.stocGradAscent1(np.array(TrainSet),TrainLabel,2000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorsum = 0.0
    for k in range(numTests):
        errorsum += colicTest()
    print("the average error rate is %f" %(errorsum / float(numTests)))

def main():
    multiTest()

if __name__ == '__main__':
    main()