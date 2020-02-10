import matplotlib.pyplot as plt
import matplotlib
import tree

decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle="<-")
matplotlib.rc("font", family="AR PL UKai CN")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords="axes fraction",
                             xytext=centerPt,textcoords="axes fraction",
                             va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

def zerokey(myTree):
    for key in myTree:
        return key

def getNumLeafs(myTree):  #获取叶子节点数目
    numLeafs = 0
    firstStr = zerokey(myTree)
    secondDict = myTree[firstStr]
    for key in secondDict.keys():  #枚举该属性每一个可能的元素
        if type(secondDict[key]).__name__ == 'dict': #判断类型是否为字典
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDeep(myTree): #获取树的深度
    maxDeep = 0
    firstStr = zerokey(myTree)
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDeep = 1 + getTreeDeep(secondDict[key])
        else: thisDeep = 1
        if thisDeep > maxDeep: maxDeep = thisDeep
    return maxDeep

#在父子节点之间填充文本
def PlotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDeep(myTree)
    firstStr = zerokey(myTree)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    PlotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            PlotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDeep(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

if __name__ == '__main__':
    Tree = tree.main()
    #getNumLeafs(Tree)
    #print(getNumLeafs(Tree))
    #print(getTreeDeep(Tree))
    createPlot(Tree)