from pyspark import SparkContext, SparkConf

# Import the Dataset
lNumbers = []

with open('dataset.txt', 'r') as f:
    lines = f.readlines()
    for n in lines:
    	if len(n[:-1]) > 0:
    		lNumbers.append(float(n[:-1]))

print("The locally loaded list of numbers is: ", lNumbers)

# Spark Setup

conf = SparkConf().setAppName('Somma di quadrati').setMaster('local')
sc = SparkContext(conf=conf)

# Create a parallel collection
dNumbers = sc.parallelize(lNumbers)

sumOfSquares = dNumbers.map(lambda s: s**2).reduce(lambda a, b: a + b)

print("The sum of squares is " + str(sumOfSquares))