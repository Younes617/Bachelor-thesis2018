# User manual

If you have a graph and want to test out which of the implementations is best suited time-wise. The predictGraph.py file can be used. To select the prefered machine learning method the values 1-3 can be set.


1: Linear Regression

2: Decision tree

3: Random Forest


## Steps

(1) Input graph: must be in adjacency list format. (I only tested it on graphs from DIMACS (https://www.cc.gatech.edu/dimacs10/downloads.shtml) but it should work on other graphs too.

(2) Make sure the file (predictGraph.py) is in the same folder as trainData.csv, polakData.csv, jainData.csv and fivesum.csv because it trains the model using those data.

(3) Use the command: python predictGraph.py *1-3* *file.graph*

Example: python predictGraph.py 1 graphs/delaunay_n10.graph

###Output


Polak: if Polaks algorithm is optimal

Jain: if Jain & Adtani's algorithm is optimal
