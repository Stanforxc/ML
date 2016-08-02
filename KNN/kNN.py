from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

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

group,labels = createDataSet()
print classify0([0,0],group,labels,3)