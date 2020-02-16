import numpy as np
import math
import random
def loadDataSet(): #创建数据,返回一组包含五条留言的词条和对应的词条属性,0表示正常言论,1表示侮辱性文字
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def CreateDictionary(dataSet): #创建一个祠表
    Dictionary = set([])
    for word in dataSet:
        Dictionary = Dictionary | set(word)
    return list(Dictionary)

def SetOfWordstoVec(dictionary,inputSet): #将句子在词表中转换为对应的向量
    returnVec = [0]*len(dictionary)
    for word in inputSet:
        if word in dictionary:
            returnVec[dictionary.index(word)] = 1
        else:
            print("the word: %s is not in mu Vocabulary!" % word)
    return returnVec

def BagOfWordstoVec(dictionary,inputSet): #词袋模型，一个词语可出现多次
    returnVec = [0]*len(dictionary)
    for word in inputSet:
        if word in dictionary:
            returnVec[dictionary.index(word)] += 1
        else:
            print("the word: %s is not in mu Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory): #训练函数
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)  #p(w_i,c_0)
    p1Num = np.ones(numWords)  #p(w_i,c_1)
    p0Denom = 2.0; p1Denom = 2.0 #该类别的总词数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix)
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix)
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive #p0,p1的各个属性的概率向量,侮辱性语句的占比

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass) #II(p(w_i,c1))
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass) #II(p(w_i,c0))
    if p1 > p0:
        return 1
    else:
        return 0

def textParse(bigString): #将文本划分为含有独立单词的列表
    import re
    listOfTokens = re.split(r'\W*',bigString)
    returnlist = []
    for word in listOfTokens:
        if len(word) > 2: returnlist.append(word.lower())
    return returnlist

def spamTest():
    dataset = []
    dataclass = []
    for i in range(1,26): #分别读如26个正反语句,存入docList和fullText,将他们的性质存入classList
        wordList = textParse(open('email/spam/%d.txt' % i,encoding='ISO-8859-1').read())
        dataset.append(wordList)
        dataclass.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding='ISO-8859-1').read())
        dataset.append(wordList)
        dataclass.append(0)
    dictionary = CreateDictionary(dataset)  #字典
    trainingSet = list(range(50))  #创建一个0-49的列表
    testSet = []
    for i in range(10): #从中随机选出10个,作为测试集的下标
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []

    for docIndex in trainingSet: #其余的作为训练集的下标
        trainMat.append(SetOfWordstoVec(dictionary,dataset[docIndex]))
        trainClasses.append(dataclass[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = SetOfWordstoVec(dictionary,dataset[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != dataclass[docIndex]:
            errorCount += 1
    print(errorCount)
    print(len(testSet))
    print(f'the error rate is {float(errorCount)/len(testSet)}')
def main():
    '''
    dataset,classes = loadDataSet()
    dictionary = CreateDictionary(dataset)
    trainMat = []
    for postinDoc in dataset:
        trainMat.append(SetOfWordstoVec(dictionary,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,classes)
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(SetOfWordstoVec(dictionary,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(SetOfWordstoVec(dictionary, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    '''
    spamTest()
if __name__ == '__main__':
    main()