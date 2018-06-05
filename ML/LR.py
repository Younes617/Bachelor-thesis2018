# Name: Younes Ouazref
# StudentID: 10732519
# University of Amsterdam
# 
# Date: 28-05-2018
#
# Description: 	Small script to use Linear Regression to predict 
# 				which implementation of the triangle counting algorithm
# 				gives the best performance. The average accuracy is given as 
# 				output.



import pandas as pd
import numpy as np
# import seaborn as sns
import csv

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from random import randint


# Read in the data of the algorithms and the graphs
file1 = 'polakData.csv'
with open(file1) as fp1:
    data1 = pd.read_csv(fp1)

file2 = 'jainData.csv'
with open(file2) as fp2:
    data2 = pd.read_csv(fp2)

file3 = 'fivesum.csv'
with open(file3) as fp3:
    data3 = pd.read_csv(fp3)


# Add the graph data to the algo data
polakData = pd.concat([data1, data3], axis=1, join_axes=[data1.index])
jainData = pd.concat([data2, data3], axis=1, join_axes=[data2.index])


features = ['NumVertices', 'NumEdges', 'Median', 'LowestValue', 'HighestValue', 'LowerQ', 'UpperQ', 'StandardDev']


X1 = polakData[features]
y1 = polakData['Time']

X2 = jainData[features]
y2 = jainData['Time']


polak = "Polak"
jain = "Jain"

accList = []


# Run the model 5 times and take the average accuracy
runs = 10
for a in range(0, runs):

	polakTime = []
	jainTime = []

	compareList = []
	testAlgo = []


	r = randint(0, 1000)

	X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=r)
	X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=r)


	linreg1 = LinearRegression()
	linreg1.fit(X1_train, y1_train)

	linreg2 = LinearRegression()
	linreg2.fit(X2_train, y2_train)


	# Here the model is run on the formula that the linear regression has made
	for j in range(0, len(X1_test.values)):
		test1 = linreg1.intercept_
		for i in range(0, len(X1_test.values[j])):
			test1 += (linreg1.coef_[i] * X1_test.values[j][i])
		polakTime.append(test1)


	for m in range(0, len(X2_test.values)):
		test2 = linreg2.intercept_
		for n in range(0, len(X2_test.values[m])):
			test2 += (linreg2.coef_[n] * X2_test.values[m][n])
		jainTime.append(test2)

	# Here the predicted times of the two algorithms are being compared and
	# the one with the lowest predicted execution time is the optimal one.
	for c in range(0, len(polakTime)):
		if polakTime[c] < jainTime[c]:
			compareList.append(polak)
		else:
			compareList.append(jain)

	# Here the fastest algo is taken from the real times they took.
	for d in range(0, len(y1_test)):
		if y1_test.values[d] < y2_test.values[d]:
			testAlgo.append(polak)
		else:
			testAlgo.append(jain)


	print "Prediction: " + str(compareList)
	print "Real:       " + str(testAlgo)
	# https://stackoverflow.com/questions/38877301/how-to-calculate-accurady-based-on-two-lists-python
	acc = sum(1 for x,y in zip(compareList, testAlgo) if x == y) / float(len(compareList))
	accList.append(acc)
	print "Accuracy: " + str(acc * 100) + "%" 

avgAcc = sum(accList) / float(len(accList))
print "Accuracy after " + str(runs) + " runs is: "
print str(avgAcc * 100) + "%" + " using Linear Regression"

