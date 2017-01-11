#coding:utf-8
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,creatCent=randCent):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = creatCent(dataMat,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True #当任意一个点的族分配发生改变时
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k): #求簇内平均值
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

"""
def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while(len(centList)<k):
        lowestSSE = inf
        for i in range(len(centList)):
"""



dataMat = mat(loadDataSet('testSet.txt'))
myCentroids,clustAssing = kMeans(dataMat,4)

print "----------------------"
print myCentroids
print clustAssing