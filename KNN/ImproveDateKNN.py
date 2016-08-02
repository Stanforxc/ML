#coding:utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator

#识别
def classifyPerson():
    result = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input(
        "percentage of time spent playing video games?"
    ))
    ffMiles = float(raw_input(
        "frequent flier miles earned per year?"
    ))
    iceCream = float(raw_input(
        "liters of ice cream consumed per year?"
    ))
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierrResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person:",result[classifierrResult-1]

#KNN分类算法
def classify0(inX, dataSet ,labels , k):
    dataSetSize = dataSet.shape[0]  #The matrix's size. shape[0] means the size of row ,shape[1] means the size of column
    diffMat = tile(inX,(dataSetSize,1)) - dataSet  #tile is a numpy function
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


#测试分类器效果
def datingClassTest():
    hoRatio = 0.1 #10%的测试数据
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is %d"%(classifierResult,datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is %f"%(errorCount/float(numTestVecs))



#归一化  newValue=(oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)  #选取当列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = dataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


#整理数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

classifyPerson()

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
'''
