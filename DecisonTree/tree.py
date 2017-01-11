#coding:utf-8
from math import log
import operator
#计算给定数据集的香农熵,度量数据集的无序程度
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

#创建数据集
def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures  =len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    baseInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > baseInfoGain):
            baseInfoGain = infoGain
            bestFeature = i
    return bestFeature

#返回次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#构建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]#属性
    if classList.count(classList[0]) == len(classList):#类别完全相同，则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:    #遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最好的特征
    beatFeatLabel = labels[bestFeat]
    myTree = {beatFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[beatFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#使用决策树进行分类
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'Dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
        else:
            classLabel = secondDict[key]
    return classLabel

#数据持久化，pickle模块
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


"""
myDat,labels = createDataSet()
tmpLabels = labels[:]
myTree =  createTree(myDat,labels)
print classify(myTree,tmpLabels,[1,0])
storeTree(myTree,"classifierStorage.txt")
print grabTree('classifierStorage.txt')
"""

#对隐形眼镜进行分类
def lensClassify():
    fr = open('lenses.txt')
    lenses = [inst.split('\t') for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    tmplensesLabels = lensesLabels[:]
    lensesTree = createTree(lenses,lensesLabels)
    classlabel =  classify(lensesTree,tmplensesLabels,["young","myope","no","normal"])
    print classlabel

lensClassify()