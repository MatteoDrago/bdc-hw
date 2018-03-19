from pyspark import SparkContext, SparkConf

# Import the Dataset
lNumbers = []

with open('dataset.txt', 'r') as f:
    lines = f.readlines()
    for n in lines:
    	if len(n[:-1]) > 0:
    		lNumbers.append(float(n[:-1]))

print lNumbers

# Spark Setup
conf = SparkConf().setAppName('Preliminaries')
sc = SparkContext(conf=conf)

# Create a parallel collection
dNumbers = sc.parallelize(lNumbers)

sumOfSquares = dNumbers.map(lambda x: x*x).reduce(lambda x, y: x+y)
print 'The sum of squares:', sumOfSquares
