from pyspark import SparkContext, SparkConf

# Import the Dataset
lNumbers = []
with open('dataset.txt', 'r') as f:
    lines = f.readlines()
    for n in lines:
    	if len(n[:-1]) > 0:
    		lNumbers.append(float(n[:-1]))


print lNumbers
