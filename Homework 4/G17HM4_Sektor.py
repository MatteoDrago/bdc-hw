from pyspark import SparkConf, SparkContext
from pyspark.ml.linalg import Vectors
import random
import time
import numpy as np


k=5
numBlocks = 2
datafile = 'test-datasets/vecs-50-10000.txt'

#Spark configuration
config = SparkConf().setAppName('Homework 4')
sc = SparkContext(conf=config)

#Utilities functions
#K-Center fast implementation with Fartherst-Traversal First
def kcenter(P, k):
    #P_minus_S = [p for p in P]
    idx_rnd = random.randint(0, len(P)-1)
    S = [P[idx_rnd]]
    dist_near_center = [Vectors.squared_distance(P[i], S[0]) for i in range(len(P))]

    for i in range(k-1):    
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        S.append(P[new_center_idx])

        for j in range(len(P)):
            if j != new_center_idx:
                dist = Vectors.squared_distance(P[j], S[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
            else:
                dist_near_center[j] = 0
    return S

#Sequential for diversity maximization
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
    coreset = distRDD.mapValues(lambda x: getMatrix(x)).mapValues(lambda x: kcenter(x,k)).flatMap(lambda x: x[1]).collect()
    print("Time taken for coreset construction: " + str(time.time()-before))
    
    # Extract k-Points maximal diversity on a single reducer
    before = time.time()
    list_max_diversity_output = runSequential(coreset,k)
    print("Time taken for completing sequential algorithm: " + str(time.time()-before))

    return list_max_diversity_output



def measure(pointlist):
    cumsum = 0
    i=0
    for point_one in pointlist:
        for point_two in pointlist:
            if point_one != point_two:
                cumsum += np.linalg.norm(point_one-point_two, 2)
                i = i+1
    return cumsum/i



# Import the Dataset and Define the Variables
pointsrdd = sc.textFile(datafile)\
    .map(lambda row : Vectors.dense([float(num_str) for num_str in row.split(' ')]))

#Solve the max-diversity problem
solution = runMapReduce(pointsrdd,k,numBlocks)
score = measure(solution)
print("the average distance among the solution points is: " + str(score))



