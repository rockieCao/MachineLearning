#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Kaggle_101_DigitalRecognizer SVM'
"""

from numpy import *
from sklearn.svm import SVC
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

def classify(trainData, trainLabel, testData):
    svc = SVC(C=5.0)#default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    svc.fit(trainData, ravel(trainLabel))
    testLabel = svc.predict(testData)
    saveASCsv(testLabel, 'svc_c5.csv')
    return testLabel

def test():
    trainData, trainLabel = loadTrainingData('train.csv')
    testData = loadTestData('test.csv')
    testLabel = loadTestLabel('rf_benchmark.csv')
    
    result = classify(trainData, trainLabel, testData)
    (m,n) = testData.shape
    for i in range(m):
        if result[i]!=testLabel[0,i]:
            errCnt += 1
        if i%5000==0:
            print('processed %d lines'%i)
    print('error count: ', errCnt)
    print('error rate: ', errCnt/float(m))

if __name__ == '__main__':
    from time import clock
    start = clock()
    test()
    finish = clock()
    print('run time: ', start-finish)

