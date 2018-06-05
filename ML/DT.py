# Name: Younes Ouazref
# StudentID: 10732519
# University of Amsterdam
# 
# Date: 28-05-2018
#
# Description: 	Small script to train a Decision tree classifier to predict 
# 				which implementation of the triangle counting algorithm
# 				gives the best performance. The average accuracy is given as 
# 				output.

import pandas as pd
import numpy as np
# import seaborn as sns
import csv

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


#  Load in the data
file1 = 'trainData.csv'
with open(file1) as fp1:
    df = pd.read_csv(fp1)

# Select the features to use
features = ['NumVertices', 'NumEdges', 'Median', 'LowestValue', 'HighestValue', 'LowerQ', 'UpperQ', 'StandardDev']
# Select target value
target = 'Algo'

X = df[features]
y = df[target]

avgAcc = []

runs = 10
# Run the classifier 10 times and average the accuracy
for a in range(0, runs):
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	
	dtree = DecisionTreeClassifier()
	
	dtree.fit(X_train, y_train)

	print "Prediction: " + str(dtree.predict(X_test))

	print "Real:       " + str(y_test.values)

	acc = accuracy_score(y_test, dtree.predict(X_test))
	avgAcc.append(acc)
	print "Accuracy:   " + str(acc * 100) + "%"


print "Accuracy after " + str(runs) + " is:" 
print str((sum(avgAcc) / len(avgAcc)) * 100) + "%  using Decision Tree" 