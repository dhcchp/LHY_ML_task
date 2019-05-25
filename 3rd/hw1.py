# -*- coding: utf-8 -*-
# referred: https://blog.csdn.net/Crafts_Neo/article/details/90356621

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

path = "./data/"

train = pd.read_csv(path + 'train.csv',engine = 'python', encoding = 'utf-8')
test  = pd.read_csv(path + 'test.csv' ,engine = 'python', encoding = 'gbk')

train = train[train['observation'] == 'PM2.5']
# print(train)
test = test[test['AMB_TEMP'] == 'PM2.5']
# print(test)
train = train.drop(['Date','stations','observation'],axis=1)
# print(train)
test_x = test.iloc[:,2:]
#print(test_x)

train_x = []
train_y = []

for i in range(15):
        x = train.iloc[:,i:i+9]
        x.columns = np.array(range(9))
        y=train.iloc[:,i+9]
        y.columns = np.array(range(1))
        train_x.append(x)
        train_y.append(y)

train_x = pd.concat(train_x)
train_y = pd.concat(train_y)

train_y = np.array(train_y,float)
test_x  = np.array(test_x,float)
#print(test_x)
#print(train_y)
ss = StandardScaler()

#print(ss)
print("-----")
ss.fit(train_x)
#print(ss)
train_x = ss.transform(train_x)

ss.fit(test_x) # what's the meaning?
test_x = ss.transform(test_x)

def r2_score(y_true,y_predict):
    MSE = np.sum((y_true-y_predict)**2)/len(y_true)
    return 1-MSE/np.var(y_true)

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0], \
                "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self,X_train,y_train,eta = 0.01, n_iters = 1e4):
        assert X_train.shape[0] == y_train.shape[0], \
                "the size of X_train must be equal to the size of y_train"

        def J(theta,X_b,y):
            try:
                return np.sum((y-X_b.dot(theta))**2)/len(y)
            except:
                return float("inf")

        def dJ(theta,X_b,y):
            return X_b.T.dot(X_b.dot(theta)-y)*2./len(y)

        def gradient_descent(X_b,y,initial_theta,eta,n_iters = 1e4,epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(theta,X_b,y)
                last_theta = theta
                theta = theta-eta*gradient
                if (abs(J(theta,X_b,y)-J(last_theta,X_b,y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self,X_predict):
        assert self.intercept_ is not None and self.coef_ is not None,\
                "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
                "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict),1)),X_predict])
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return

LR = LinearRegression().fit_gd(train_x,train_y)
LR.score(train_x,train_y)
result = LR.predict(test_x)

sampleSubmission = pd.read_csv(path+'sampleSubmission.csv',engine='python',encoding='gbk')
sampleSubmission['value'] = result
sampleSubmission.to_csv(path+'result.csv')

