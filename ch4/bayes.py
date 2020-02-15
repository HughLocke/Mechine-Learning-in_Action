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

def main():
    dataset,classes = loadDataSet()
    dictionary = CreateDictionary(dataset)

    print(SetOfWordstoVec(dictionary,dataset[0]))


if __name__ == '__main__':
    main()