# Import Packages
from pyspark import SparkConf, SparkContext
from pyspark.ml.linalg import Vectors
import numpy as np
import random
import time

########################################
# Build the Functions
########################################

# Load the Dataset
def readVectorsSeq(filename):
    file = open(filename, 'r')
    vector_list = []
    for row in file.readlines():
        vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
    return vector_list

# K-Center
def kcenter(P, k):
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

# K-Means++
def kmeansPP(P, k):
    idx_rnd = random.randint(0, len(P)-1)
    S = [P[idx_rnd]]
    dist_near_center = [Vectors.squared_distance(P[i], S[0]) for i in range(len(P))]

    for i in range(k-1):
        
        weights = dist_near_center/np.sum(dist_near_center)
        idx = np.random.choice(range(len(P)),p=weights)
        S.append(P[idx])

        for j in range(len(P)):
            if j != idx:
                dist = Vectors.squared_distance(P[j], S[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
            else:
                dist_near_center[j] = 0 # this assures that in dist_near_center we consider just distances of P minus S
    return S

# K-Means Objective Function
def kmeansObj(P, C):
    obj = 0
    related_center_idx = []
    
    for p in P:
        
        # Find the closest Center
        dist_min = Vectors.squared_distance(p, C[0])
        for c in C:
            dist = Vectors.squared_distance(p, c)
            
            if dist < dist_min:
                dist_min = dist
                
        # Update the Objective Function   
        obj += dist_min / len(P)
                
    return obj

########################################
# Use the Functions
########################################

# Load the Dataset
vector_list = readVectorsSeq('test-datasets/vecs-50-10000.txt')

# Get the Parameters k and k1 from the User
print('Insert k and k1 such that k < k1.')
k = int(input('Digit k:'))
k1 = int(input('Digit k1:'))
print()

# Get K-Center Time Performance
#P = [p for p in vector_list] # Copy the Points
t0 = time.time()
C_kcenter = kcenter(vector_list, k)
t1 = time.time()

# Print the Results
print('K-Center Result:')
print('Elapsed Time :', t1-t0, 's')
print()

# Get k Centers from K-Means++
#P = [p for p in vector_list]
C_kmeansPP = kmeansPP(vector_list, k)

# Get the Objective Function 
obj = kmeansObj(vector_list, C_kmeansPP)

# Print the Results
print('K-Center Result:')
print('Objective Function from K-Means++ with k :', obj)
print()

# Get k1 Centers from K-Center
C_kcenter_1 = kcenter(vector_list, k1)
C_kmeansPP_1 = kmeansPP(C_kcenter_1, k)
obj_1 = kmeansObj(vector_list, C_kmeansPP_1)

# Print the Results
print('K-Center Result:')
print('Objective Function from K-Means++ with k1 :', obj_1)
print()