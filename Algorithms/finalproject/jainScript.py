# Name: Younes Ouazref
# StudentID: 10732519
# University of Amsterdam
# 
# Date: 28-05-2018
# 
# Description: 	Script to run Jain et al. implementation on multiple graphs
# 				and to collect the data.


import subprocess
import csv


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

# features to use for the model.
features = []
features.append(['NumVertices', 'NumEdges', 'Time', 'Algo'])

checked = False
data = []

reruns = 5

for x in range(0, len(graphs)):
	
	for y in range(0, reruns - 1):
		print y
		# Collect the data for multiple graphs at the same time on DAS-5 (Supercomputer located in the Netherlands)
		p = subprocess.Popen("prun -np 1 -native '-C K40 --gres=gpu:1' ./triangleCounter data/" + graphs[x], stdout=subprocess.PIPE, shell=True)

		(output, err) = p.communicate()

		newOutput = output.splitlines()
		times = []

		# Here I save the time every GPU kernel call took. 
		for y in range(0, len(newOutput)):
			# Store the names of the values once at the start.
			if checked == False:
				data.append(newOutput[y].split(":", 1)[0])
			# Check if the char ':' is in the line (Only my comments in his 
			# implementation contains ':')
			if newOutput[y].find(":") != -1:
				times.append([data[y], newOutput[y].split(":", 1)[1].strip().split()[0]])


		totalGpuTimes = []
		for a in times:
			# The vertices and edges are taken from the top of the file.
			# if a[0] == 'Num vertices':
			# 	numVer = a[1]
			# elif a[0] == 'Num edges':
			# 	numEdges = a[1] 
			if a[0] == 'Total GPU time':
				totalGpuTimes.append(float(a[1]))



	with open("data/" + graphs[x]) as f:
		# Only interested in the first line which contains vertices and edges.
		first_line = f.readline()

	Gsize = first_line.split()

	# Here the right values are taken from the output.
	numVer = Gsize[0]
	numEdges = Gsize[1]
	algo = "Jain"
	# Get average time of the runs
	totalGpuTime = sum(totalGpuTimes) / len(totalGpuTimes)
	# Add the values as features
	features.append([numVer, numEdges, totalGpuTime, algo])
	checked = True


with open('jainData.csv', 'w') as fp:
	a = csv.writer(fp, delimiter=',')
	a.writerows(features)
