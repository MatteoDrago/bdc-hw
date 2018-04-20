# Import Packages
from pyspark import SparkConf, SparkContext
from pyspark.ml.linalg import Vectors
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
    P_minus_S = [p for p in P]
    idx_rnd = random.randint(0, len(P)-1)
    S = [P[idx_rnd]]
    P_minus_S.pop(idx_rnd)
    related_center_idx = [idx_rnd for i in range(len(P))]
    dist_near_center = [Vectors.squared_distance(P[i], S[0]) for i in range(len(P))]

    for i in range(k-1):    
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation

        S.append(P[new_center_idx])
        P_minus_S.remove(P[new_center_idx])

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
    P_minus_S = [p for p in P]
    idx_rnd = random.randint(0, len(P)-1)
    S = [P[idx_rnd]]
    P_minus_S.pop(idx_rnd)
    related_center_idx = [idx_rnd for i in range(len(P))]
    dist_near_center = [Vectors.squared_distance(P[i], S[0]) for i in range(len(P))]

    for i in range(k-1):    
        sum_dist = sum([d**2 for d in dist_near_center])
        probs = [d**2 / sum_dist for d in dist_near_center]
        cum_probs = [sum(probs[:i+1]) for i in range(len(P))]
        coin = random.random()
        cum_probs_minus_coin = [abs(cum_probs[j]-coin) for j in range(len(P))]
        new_center_idx = min(enumerate(cum_probs_minus_coin), key=lambda x: x[1])[0] # argmin operation
        
        # Append the New Center
        S.append(P[new_center_idx])
        P_minus_S.remove(P[new_center_idx])
        
        # Update the Distances and the Clusters
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
k = input('Digit k:')
k1 = input('Digit k1:')
print()

# Get K-Center Time Performance
P = [p for p in vector_list]
t0 = time.time()
C_kcenter = kcenter(P, k)
t1 = time.time()

# Print the Results
print('K-Center Result:')
print('Elapsed Time :', t1-t0, 's')
print()

# Get k Centers from K-Means++
P = [p for p in vector_list]
C_kmeansPP = kmeansPP(P, k)

# Get the Objective Function 
obj = kmeansObj(P, C_kmeansPP)

# Print the Results
print('K-Center Result:')
print('Objective Function from K-Means++ with k :', obj)
print()

# Get k1 Centers from K-Center
C_kcenter_1 = kcenter(P, k1)
C_kmeansPP_1 = kmeansPP(C_kcenter_1, k)
obj_1 = kmeansObj(P, C_kmeansPP_1)

# Print the Results
print('K-Center Result:')
print('Objective Function from K-Means++ with k1 :', obj_1)
print()