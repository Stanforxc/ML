#coding:utf-8
import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(u'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(u'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

#在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]+cntrPt[0])/2.0
    yMid = (parentPt[1]+cntrPt[1])/2.0
    createPlot.ax1.text(xMid,yMid.txString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    #未完，画图好无聊。。。。。

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondeDict = myTree[firstStr]
    for key in secondeDict.keys():
        if type(secondeDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondeDict[key])
        else:
            numLeafs += 1
    return numLeafs

#获取数的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'Dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
    if thisDepth > maxDepth:
        maxDepth = thisDepth
    return maxDepth

#用于测试
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

myTree = retrieveTree(0)
print getTreeDepth(myTree)
print getNumLeafs(myTree)