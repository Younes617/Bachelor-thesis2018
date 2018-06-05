# Name: Younes Ouazref
# StudentID: 10732519
# University of Amsterdam
# 
# Date: 28-05-2018
# 
# Description: 	Script that predicts the algorithm which works best for a type of graph.
# 				Input should be a graph in adjacency list format.


import subprocess
import csv
import pandas as pd
import numpy as np
import sys
import os

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from random import randint


# Give graph file through commandline like: ./program file.graph


if len(sys.argv) < 2:
    print "Use the program like this: python predictGraph.py *value from 1-3* *file.graph*"
    sys.exit(0)

ml = int(sys.argv[1])



# If first arg is not an integer between 1 and 3
if not (1 <= int(ml) <= 3):
	print "First arg must be be 1, 2 or 3"
	sys.exit(0)



graph = sys.argv[2]

if not os.path.isfile(graph):
	print "File given as second arg doesn't exist"
	sys.exit(0)


with open(graph) as f:
	# Only interested in the first line which contains vertices and edges.
    first_line = f.readline()

Gsize = first_line.split()


numVer = Gsize[0]
numEdges = Gsize[1]

skippedFirst = False # First line contains vertices and edges info

fp = open(graph,"rb")
med = []
inputFeatures = []
for d in fp.readlines():
	if skippedFirst:
		f1 = d.split()
		med.append(len(f1))
	skippedFirst = True

med.sort()

half = len(med)/2
upper = med[half:]
lower = med[:half]

inputFeatures.append([float(numVer), float(numEdges), float(np.median(med)), float(min(med)), float(max(med)),
					 float(np.median(lower)), float(np.median(upper)), float(np.std(med))])

fp.close()




if ml == 1:
	print "Machine learning method: Linear Regression"
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

	r = randint(0, 1000)

	X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=r)
	X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=r)


	linreg1 = LinearRegression()
	linreg1.fit(X1_train, y1_train)

	linreg2 = LinearRegression()
	linreg2.fit(X2_train, y2_train)

	polak = linreg1.predict(inputFeatures)
	jain = linreg2.predict(inputFeatures)

	if polak < jain:
		print "Prediction: Polak"
	else:
		print "Prediction: Jain"



else: 
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

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)


	if ml == 2:
		print "Machine learning method: Decision Tree"

		dtree = DecisionTreeClassifier()
		dtree.fit(X_train, y_train)
		print "Prediction: " + str(dtree.predict(inputFeatures)[0])

	elif ml == 3:
		print "Machine learning method: Random Forest"
		clf = RandomForestClassifier()
		clf.fit(X_train, y_train)
		print "Prediction: " + str(clf.predict(inputFeatures)[0])