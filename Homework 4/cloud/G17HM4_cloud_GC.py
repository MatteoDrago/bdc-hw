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
def runMapReduce(pointsrdd, k, numBlocks):
    
    # Partitioning
    blocks = pointsrdd.glom()
    
    # Extract k-Points using Farthest-First Traversal algorithm
    centers = blocks.map(lambda p: farthest_first_traversal(p,k))
    
    # Gathering the Vectors
    t0 = time.time()
    coreset = [y for c in centers.collect() for y in c]
    t1 = time.time()
    coreset_time = t1-t0
    
    # Results
    t0 = time.time()
    results = runSequential(coreset, k)
    t1 = time.time()
    result_time = t1-t0
    
    return results, coreset_time, result_time

# Measure
def measure(pointslist):
    dist_avg = 0.
    n_points = len(pointslist)
    for i in range(n_points):
        for j in range(i):
            dist_avg += np.sqrt(Vectors.squared_distance(pointslist[i], pointslist[j]))
    dist_avg = dist_avg / (n_points * (n_points-1) / 2)
    
    return dist_avg

########################################
# Use the Functions
########################################

# Spark Setup
conf = SparkConf().setAppName('HW4').setMaster('local')
sc = SparkContext(conf=conf)

# Import the Dataset
datafile = sys.argv[-1]
k = [i for i in range(2,6)]
k[0] = 2
numBlocks = [18, 36, 54, 72, 90]
k_min, k_max = np.min(k), np.max(k)
numBlocks_min, numBlocks_max = np.argmin(numBlocks), np.argmax(numBlocks) # 0 and 4

# Print Info of the Grid Search
print()
print('Grid Search:')
print('- numBlocks Search:', numBlocks)
print('- K Search:', k)
print()

# Define Variables
inputrdd_list = [[[] for i in range(len(k))] for j in range(len(numBlocks))]
coreset_times = np.zeros((len(numBlocks), len(k)))
result_times = np.zeros((len(numBlocks), len(k)))
objs = np.zeros((len(numBlocks), len(k)))

inputrdd = sc.textFile(datafile).map(lambda row : Vectors.dense([float(num_str) for num_str in row.split(' ')]))

# Compute Variables
for j in range(len(numBlocks)):
    inputrdd.repartition(numBlocks[j]) \
            .cache()
    for i in range(len(k)):
        print('numBlocks =', numBlocks[j], ', K =', k[i])

        # Computations                        
        results, coreset_times[j,i], result_times[j,i] = runMapReduce(inputrdd, k[i], numBlocks[j])
        objs[j,i] = measure(results)

# Saving Metric Files
np.save('out/GC/k', k)
np.save('out/GC/numBlocks', numBlocks)
np.save('out/GC/coreset_times', coreset_times)
np.save('out/GC/result_times', result_times)
np.save('out/GC/objs', objs)
        