# Name: Younes Ouazref
# StudentID: 10732519
# University of Amsterdam
# 
# Date: 28-05-2018
# 
# Description: 	Script to run Polaks implementation on multiple graphs
# 				and to collect the data.


import subprocess
import csv


graphs = ["citationCiteseer.bin",
		"coAuthorsCiteseer.bin",
		"coAuthorsDBLP.bin",
		"coPapersCiteseer.bin",
		"coPapersDBLP.bin",
		"kron_g500-simple-logn16.bin",
		"kron_g500-simple-logn17.bin",
		"kron_g500-simple-logn18.bin",
		"kron_g500-simple-logn19.bin",
		"kron_g500-simple-logn20.bin",
		"kron_g500-simple-logn21.bin",
		"rgg_n_2_15_s0.bin",
		"rgg_n_2_16_s0.bin",
		"rgg_n_2_17_s0.bin",
		"rgg_n_2_18_s0.bin",
		"rgg_n_2_19_s0.bin",
		"rgg_n_2_20_s0.bin",
		"rgg_n_2_21_s0.bin",
		"rgg_n_2_22_s0.bin",
		"rgg_n_2_23_s0.bin",
		"rgg_n_2_24_s0.bin"]


# features to use for the model.
features = []
features.append(['NumVertices', 'NumEdges' ,'Time', 'Algo'])

checked = False
data = []

reruns = 5


for x in range(0, len(graphs)):
	for y in range(0, reruns - 1):
		print y
		# Collect the data for multiple graphs at the same time on DAS-5 (Supercomputer located in the Netherlands)
		p = subprocess.Popen("prun -np 1 -native '-C K40 --gres=gpu:1' ./main.e data/" + graphs[x], stdout=subprocess.PIPE, shell=True)

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


		# NumVer and numEdges are gotten like this because Polaks implementation
		# uses binary files so this is more convenient and easy.
		totalGpuTimes = []
		for a in times:
			if a[0] == 'Num vertices':
				numVer = a[1]		
			elif a[0] == 'Num edges':
				numEdges = a[1] 
			if a[0] == 'Total GPU time':
				totalGpuTimes.append(float(a[1]))


	# with open("data/" + graphs[x]) as f:
	# 	# Only interested in the first line which contains vertices and edges.
	# 	first_line = f.readline()

	# Gsize = first_line.split()
	# numVer = Gsize[0]
	# numEdges = Gsize[1]

	algo = "Polak"
	totalGpuTime = (sum(totalGpuTimes)) / len(totalGpuTimes)


	# Add the values as features
	features.append([numVer, numEdges, totalGpuTime, algo])
	checked = True


with open('polakData.csv', 'w') as fp:
	a = csv.writer(fp, delimiter=',')
	a.writerows(features)
