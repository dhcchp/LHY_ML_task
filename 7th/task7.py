import pdb
import pandas as pd
import numpy as np

#pdb.set_trace()
DATA=pd.read_csv('watermelon_3a.csv')

def ShannonEnt(data):
	n,m = data.shape
	label = data.iloc[:,-1]
	uniq_label=np.unique(label)
	shannonEnt = 0.0
	for i in uniq_label:
		xi=np.where(label==i)
		pi=np.shape(xi)[1]/n
		shannonEnt -= pi*np.math.log(pi,2)
	return shannonEnt

def Ent(Y):
	n = len(Y)
	uniq_y = np.unique(Y)
	ent = 0.0
	for i in uniq_y:
		yi = np.where(Y==i)
		pi = np.shape(yi)[1]/n
		ent -= pi*np.math.log(pi,2)
	return ent

def CondEnt(data,x=2):
	n,m=data.shape
	X=data.iloc[:,x]
	Y=data.iloc[:,-1]
	X_uiq = np.unique(X)
	condent = 0.0
	for i in X_uiq:
		idx = np.where(X==i)
		xi = X.iloc[idx]
		yi = Y.iloc[idx]
		pxi = np.shape(idx)[1]/n
		hyx = Ent(yi)
		condent = pxi*hyx
	return condent

def CalcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1

	shannonEnt = 0.0
	for key in labelCounts:
		pro = float(labelCounts[key]) / numEntries
		shannonEnt -= pro* np.math.log(pro,2)
	return shannonEnt

tt=Ent(DATA.iloc[:,-1])
print(tt)

CondEntropy=[]
n,m=DATA.shape
for i in range(m-1):
	CondEntropy.append(CondEnt(DATA,i))
print(CondEntropy)

#pdb.set_trace()
print(CalcShannonEnt(np.array(DATA)))

