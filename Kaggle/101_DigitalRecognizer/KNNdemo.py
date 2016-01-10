#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Kaggle_101_DigitalRecognizer kNN'
"""

from numpy import *
import csv

def loadTrainingData(file):
    rows=[]
    with open(file) as f:
        lines = csv.reader(f)
        for line in lines:
            rows.append(line)
    rows.remove(rows[0]) #remove header
    rows = array(rows)
    label = rows[:,0]
    data = rows[:,1:]
    return binarize(integerize(data)), integerize(label)

def loadTestData(file):
    rows = []
    with open(file) as f:
        lines = csv.reader(f)
        for line in lines:
            rows.append(line)
    rows.remove(rows[0])
    data = array(rows)
    return binarize(integerize(data))

def loadTestLabel(file):
    rows = []
    with open(file) as f:
        lines = csv.reader(f)
        for line in lines:
            rows.append(line)
    rows.remove(rows[0])
    label = array(rows)
    return integerize(label[:,1])

def integerize(data):
    mx = mat(data)
    (m,n) = mx.shape
    newData = zeros((m,n))
    for i in range(m):
        for j in range(n):
            newData[i,j] = int(mx[i,j])
    return newData

def binarize(data):
    m,n = data.shape
    for i in range(m):
        for j in range(n):
            if data[i,j] != 0:
                data[i,j] = 1
    return data

def saveASCsv(data, file):
    with open(file, 'w', newline='') as f:
        csvWriter = csv.writer(f)
        for i in data:
            tmp = []
            tmp.append(i)
            csvWriter.writerow(tmp)

# KNN classifier
def classify(testSet, trainSet, trainLabel, K):
    testSet = mat(testSet)
    trainSet = mat(trainSet)
    trainLabel = mat(trainLabel)
    trainSetN = trainSet.shape[0]

    diffMat = tile(testSet, (trainSetN, 1)) - trainSet
    sqDiffMat = array(diffMat)**2
    sqDistance = sqDiffMat.sum(axis=1)
    dist = sqDistance**0.5
    #print('dist[] : ', dist.shape)
    sortedDistIndex = dist.argsort()
    classCount = {}
    mostLikelyClass = -1
    for i in range(K):
        voteLabel = trainLabel[0, sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
        if mostLikelyClass==-1 or classCount[mostLikelyClass]<classCount[voteLabel]:
            mostLikelyClass = voteLabel
    return mostLikelyClass

def test():
    trainSet,trainLabel = loadTrainingData('train.csv')
    testSet = loadTestData('test.csv')
    testLabel = loadTestLabel('rf_benchmark.csv')
    (m, n) = testSet.shape
    resultList = []
    errCnt = 0
    for i in range(m):
        result = classify(testSet[i], trainSet[0:2000], trainLabel[0:2000], 5)
        resultList.append(result)
        #print('classified as: %d, real answer is: %d'%(result, testLabel[0,i]))
        if result != testLabel[0,i]:
            errCnt += 1
        if i%5000==0:
            print('processed %d lines'%i)
    print('error count: ', errCnt)
    print('error ratio: ', errCnt/float(m))
    saveASCsv(resultList, 'result.csv')

if __name__ == '__main__':
    from time import clock
    start = clock()
    test()
    finish = clock()
    print('run time: %fs'%(finish-start))
