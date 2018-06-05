# Name: Younes Ouazref
# StudentID: 10732519
# University of Amsterdam
# 
# Date: 28-05-2018
# 
# Description: 	Script to calculate the five number summary of the graphs used


import subprocess
import csv
import numpy as np

graphs = ["citationCiteseer.graph",
		"coAuthorsCiteseer.graph",
		"coAuthorsDBLP.graph",
		"coPapersCiteseer.graph",
		"coPapersDBLP.graph",
		"kron_g500-simple-logn16.graph",
		"kron_g500-simple-logn17.graph",
		"kron_g500-simple-logn18.graph",
		"kron_g500-simple-logn19.graph",
		"kron_g500-simple-logn20.graph",
		"kron_g500-simple-logn21.graph",
		"rgg_n_2_15_s0.graph",
		"rgg_n_2_16_s0.graph",
		"rgg_n_2_17_s0.graph",
		"rgg_n_2_18_s0.graph",
		"rgg_n_2_19_s0.graph",
		"rgg_n_2_20_s0.graph",
		"rgg_n_2_21_s0.graph",
		"rgg_n_2_22_s0.graph",
		"rgg_n_2_23_s0.graph",
		"rgg_n_2_24_s0.graph"]


answer = []

# Loop over every graph
for x in range(0, len(graphs)):
	skippedFirst = False # First line contains vertices and edges info

	fp = open("data/" + graphs[x],"rb")
	med = []
	for d in fp.readlines():
		if skippedFirst:
			f1 = d.split()
			med.append(len(f1))

		skippedFirst = True

	med.sort()
	print "len med " + str(len(med))
	print "min med " + str(min(med))
	print "max med " + str(max(med))
	print "median all " + str(np.median(med))

	half = len(med)/2
	upper = med[half:]
	lower = med[:half]

	print "lower quartile " + str(np.median(lower))
	print "upper quartile " + str(np.median(upper))
	print

	answer.append([np.median(med), min(med), max(med), np.median(lower), np.median(upper), np.std(med)])
	print


with open('fivesum.csv', 'wb') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(['Median', 'LowestValue', 'HighestValue', 'LowerQ', 'UpperQ', 'StandardDev'])
	for x in answer:
		writer.writerow(x)
