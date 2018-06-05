# Name: Younes Ouazref
# StudentID: 10732519
# University of Amsterdam
# 
# Date: 28-05-2018
# 
# Description:  This file generates a new csv file by combining the results of
#				different csv files.  


import pandas as pd
import numpy as np
import csv


# This csv file contains the number of vertces, number of edges and execution 
# time of Adam Polaks implementation on different graphs.
with open('polakData.csv', 'rb') as f1:
	reader1 = csv.reader(f1)
	your_list1 = list(reader1)

# This csv file contains the number of vertces, number of edges and execution 
# time of Manish Jain and Vashishtha Adtanis implementation on different graphs.
with open('jainData.csv', 'rb') as f2:
	reader2 = csv.reader(f2)
	your_list2 = list(reader2)

# This csv file contains the 5 number summary and standard deviation of the
# dergee distributions per graph.
with open('fivesum.csv', 'rb') as f3:
	reader3 = csv.reader(f3)
	degree = list(reader3)

your_list3 = []

# Check which algorithm has a faster execution time and use that row together
# with the five number summary and stdev as input data for the model.
for x in range(0, len(your_list1)):
	if(x != 0):
		# 2 because the time is the thirs column
		if float(your_list1[x][2]) <= float(your_list2[x][2]):
			your_list3.append(your_list1[x]+degree[x])
		else:
			your_list3.append(your_list2[x]+degree[x])
	else:
		your_list3.append(your_list1[x]+degree[x])


# Write the combined data to a new csv file.
with open('trainData.csv', 'wb') as myfile:
	wr = csv.writer(myfile, delimiter=',')
	for item in your_list3:
		wr.writerow(item)
