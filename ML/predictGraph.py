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

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Give graph file through commandline like: ./program file.graph
graph = sys.argv[1]


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

inputFeatures.append([numVer, numEdges, np.median(med), min(med), max(med), np.median(lower), np.median(upper), np.std(med)])

fp.close()


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

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

print "Prediction using Decision Tree: " + str(dtree.predict(inputFeatures)[0])
