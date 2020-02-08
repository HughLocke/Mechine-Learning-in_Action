from numpy import *
import operator

def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#dataSet:训练样本集合,labels:训练样本的分类,inX:用于分类的输入向量(一个测试数据)
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]   #样例的大小
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #将测试数据扩充为一个列表,每个元素diffMat_i = inX_i - dataSet_i,相当于将测试数据分别对每个训练数据作差
    sqDiffMat = diffMat * 2
    sqDiffMat = sqDiffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    # 分别求出测试数据对每个训练数据的欧式距离

    sortedDistIndicies = distances.argsort() #返回数组从小到大的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
    key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group,labels = createDataset()
    igroup = group
    print(classify0([0,0],group,labels,3))


