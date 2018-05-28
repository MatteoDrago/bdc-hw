# Import Packages
from pyspark import SparkConf, SparkContext
from pyspark.ml.linalg import Vectors
import numpy as np
import time
import random
import sys

########################################
# Build the Functions
########################################

# Farthest First Traversal
def farthest_first_traversal(P, k):
    idx_rnd = random.randint(0, len(P)-1)
    S = [P[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(P))]
    dist_near_center = [Vectors.squared_distance(P[i], S[0]) for i in range(len(P))]

    for i in range(k-1):    
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation

        S.append(P[new_center_idx])

        for j in range(len(P)):
            if j != new_center_idx:
                dist = Vectors.squared_distance(P[j], S[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return S

# Run Sequential
def runSequential(points, k):

    n = len(points)
    if k >= n:
        return points

    result = list()
    candidates = np.full(n , True)

    for iter in range(int(k / 2)):
        maxDist = 0.0
        maxI = 0
        maxJ = 0
        for i in range(n):
            if candidates[i] == True:
                for j in range(n):
                    d = Vectors.squared_distance(points[i], points[j])
                    if d > maxDist:
                        maxDist = d
                        maxI = i
                        maxJ = j
        result.append( points[maxI] )
        result.append( points[maxJ] )
        #print "selecting "+str(maxI)+" and "+str(maxJ)
        candidates[maxI] = False
        candidates[maxJ] = False

    if k % 2 != 0:
        s = np.random.randint(n)
        for i in range(n):
            j = (i + s) % n
            if candidates[j] == True:
                #print "selecting "+str(j)
                result.append( points[i] )
                break

    return result

# Run Map Reduce
#New functions
def getMatrix(x):
    mat = []
    for element in x:
        mat.append(element)
    return mat


def runMapReduce(pointsrdd, k, numBlocks):
    #partitions pointsrdd into numBlocks subsets
    before = time.time()
    distRDD = pointsrdd.map(lambda x: (np.random.randint(numBlocks),x)).groupByKey() #map each vector with a random index. Given the uniform distribution, with a big dataset I expect the partition to be uniform.
    
    ##k-point extraction with Fartherst-First Traversal 
    coreset = distRDD.mapValues(lambda x: getMatrix(x)).mapValues(lambda x: farthest_first_traversal(x,k)).flatMap(lambda x: x[1]).collect()
    time_coreset = time.time()-before
    
    # Extract k-Points maximal diversity on a single reducer
    before = time.time()
    list_max_diversity_output = runSequential(coreset,k)
    time_seq = time.time()-before

    return list_max_diversity_output, time_coreset, time_seq



def measure(pointlist):
    cumsum = 0
    i=0
    for point_one in pointlist:
        for point_two in pointlist:
            if point_one != point_two:
                cumsum += np.linalg.norm(point_one-point_two, 2)
                i = i+1
    return cumsum/i

########################################
# Use the Functions
########################################

# Spark Setup
conf = SparkConf().setAppName('HW4').setMaster('local[*]')
sc = SparkContext(conf=conf)

# Import the Dataset and Define the Variables
datafile = sys.argv[-1]
numBlocks_min, numBlocks_max = 2, 6
k = [i for i in range(2, 100)]
numBlocks = [i for i in range(numBlocks_min, numBlocks_max+1)]
k_min, k_max = np.min(k), np.max(k)

results = [[[] for i in range(len(k))] for j in range(len(numBlocks))]
coreset_times = np.zeros((len(numBlocks), len(k)))
result_times = np.zeros((len(numBlocks), len(k)))
objs = np.zeros((len(numBlocks), len(k)))

for i in range(len(k)):
	print('K =', k[i])
	for j in range(len(numBlocks)):
		inputrdd = sc.textFile(datafile)\
								.map(lambda row : Vectors.dense([float(num_str) for num_str in row.split(' ')]))

        # Computations                        
		results[j][i], coreset_times[j,i], result_times[j,i] = runMapReduce(inputrdd, k[i], numBlocks[j])
		objs[j,i] = measure(results[j][i])
        
# Saving Useful Files
np.save('out/Sektor/coreset_times', coreset_times)
np.save('out/Sektor/result_times', result_times)
np.save('out/Sektor/objs', objs)
np.save('out/Sektor/k', k)
np.save('out/Sektor/numBlocks', numBlocks)