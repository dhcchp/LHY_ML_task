import numpy as np
import pandas as pd
from collections import Counter
from decisionTreePlot import *

def calcShannonEnt(data):
	num = data.shape[0]
	labelCounts = Counter()
	for i in range(num):
		labelCounts[data[i][-1]] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		pro = float(labelCounts[key]) / num
		shannonEnt -= pro * np.math.log(pro,2)
	return shannonEnt

def splitDataSet(data,axis,value):
	reDataSet = []
	for featVec in data:
		if featVec[axis] == value:
			reduceFeatVec = list(featVec[:axis])
			reduceFeatVec.extend(featVec[axis+1:])
			reDataSet.append(reduceFeatVec)
	return np.array(reDataSet)

def splitContinuousDataSet(data,axis,value,direction):
	reDataSet = []
	for featVec in data:
		if direction == 0:
			if featVec[axis] > value:
				reduceFeatVec = featVec[:axis]
				reduceFeatVecList = list(reduceFeatVec)
				reduceFeatVecList.extend(featVec[axis+1:])
				reDataSet.append(reduceFeatVecList)
		else:
			if featVec[axis] <= value:
				reduceFeatVec = featVec[:axis]
				reduceFeatVecList = list(reduceFeatVec)
				reduceFeatVecList.extend(featVec[axis+1:])
				reDataSet.append(reduceFeatVecList)
	return np.array(reDataSet)

def chooseBestFeatureToSplit(data):
	bestFeature = -1
	bestInfoGain = 0.0
	baseEntropy = calcShannonEnt(data)
	numFeature = data.shape[1] -1
	for i in range(numFeature):
		featlist = [example[i] for example in data]
		if type(featlist[0]).__name__ == 'float' or type(featlist[0]).__name__ == 'int':
			sortfeatList = sorted(featlist)
			splitList = []
			for j in range(len(sortfeatList) -1):
				splitList.append((sortfeatList[j] + sortfeatList[j+1])/2.0)
			bestSplitEntropy = 10000
			slen = len(splitList)
			for j in range(slen):
				value = splitList[j]
				newEntropy = 0.0
				subDataSet0 = splitContinuousDataSet(data,i,value,0)
				subDataSet1 = splitContinuousDataSet(data,i,value,1)

				pro0 = len(subDataSet0) / len(data)
				newEntropy += pro0 * calcShannonEnt(subDataSet0)
				pro1 = len(subDataSet1) / len(data)
				newEntropy += pro1 * calcShannonEnt(subDataSet1)

				if newEntropy < bestSplitEntropy:
					bestSplitEntropy = newEntropy
					infoGain = baseEntropy - bestSplitEntropy
		else:
			uniqueVals = set(featlist)
			newEntropy = 0.0
			for value in uniqueVals:
				subDataSet = splitDataSet(data,i,value)
				pro = len(subDataSet) / float(len(data))
				newEntropy += pro * calcShannonEnt(subDataSet)
			infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classlist):
	classCount = {}
	for vote in classlist:
		if vote not in classCount.key():
			classCount[vote] = 0
		classCount[vote] += 1
	return max(classCount)

def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel: {}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

def test(dataSet,columns):
	data=np.array([example[1:] for example in dataSet])
	curTree = createTree(data,columns)
	createPlot(curTree)

if __name__ == "__main__":
	file = pd.read_csv("watermelon_3a.csv")
	title = file.keys()
	columns = file.columns.values.tolist()[1:]
	data = file.values
	test(data,columns)

