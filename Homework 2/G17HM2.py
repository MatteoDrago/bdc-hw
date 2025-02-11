# Import the Packages
import numpy as np
import time
from pyspark import SparkConf, SparkContext
import sys

config = SparkConf().setAppName('Homework 2').setMaster('local[*]')
sc = SparkContext(conf=config)

# Load the Dataset
filename = sys.argv[-1]
docs = sc.textFile(filename).cache() #Here we could have repartitioned the RDD but, since we are running the code with a modest input for our machine capabilities, we can avoid it.
N_documents = docs.count()
N_words = docs.flatMap(lambda document: document.split(' ')).count()

print()
print('INFO OF THE DATASET:')
print('N total words :', N_words)
print('N total documents :', N_documents)
print()

# Improved WordCount 1
def f1(document) :
    pairs_dict = {} # we used dictionary to store (key, value) pairs to use native data structure capabilities
    for word in document.split(' '):
        if word not in pairs_dict.keys():
            pairs_dict[word] = 1
        else :
            pairs_dict[word] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def f2(pair):
    word, occurecies = pair[0], list(pair[1])
    sum_o = 0
    for o in occurecies:
        sum_o += o
    return (word, sum_o)
 
wordcount_1 = docs.flatMap(f1) \
                .groupByKey() \
                .map(f2)

# Time meausure struct (used also for the following methods)
t0 = time.time()
n_dif_words = wordcount_1.count()
t1 = time.time()

# Print the Results
print('IMPROVED WORDCOUNT 1 RESULTS:')
print('Number of Different Words :', n_dif_words)
print('Elapsed Time :', t1-t0, 's')
print()

# Improved WordCount 2
def f3(document):
    dictionary = {}
    partitions = np.floor(np.sqrt(N_words)) - 1
    for word in document.split(' '):
        if word in dictionary.keys():
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    return [(np.random.randint(partitions), (k,dictionary[k])) for k in dictionary.keys()] # returns list of (x, (word, c_word))
    
def f4(document):
    pairs_dict = {}
    for pair in list(document[1]):
        word, count = pair
        if word in pairs_dict.keys():
            pairs_dict[word] += count
        else:
            pairs_dict[word] = count
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

wordcount_2 = docs.flatMap(f3)\
                .groupByKey()\
                .flatMap(f4)\
                .reduceByKey(lambda a,b: a+b)

t0 = time.time()
n_dif_words = wordcount_2.count()
t1 = time.time()
                
# Print the Results
print('IMPROVED WORDCOUNT 2 RESULTS:')
print('Number of Different Words :', n_dif_words)
print('Elapsed Time :', t1-t0, 's')
print()

# Improved WordCount with reducedByKey()
def f5(document) :
    pairs_dict = {}
    for word in document.split(' '):
        if word not in pairs_dict.keys():
            pairs_dict[word] = 1
        else :
            pairs_dict[word] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

 
wordcount_1_r = docs.flatMap(f5) \
                .reduceByKey(lambda accum, n: accum + n)

t0 = time.time()
n_dif_words = wordcount_1_r.count()
t1 = time.time()
                
# Print the Results
print('IMPROVED WORDCOUNT WITH reducedByKey() Result:')
print('Number of Different Words :', n_dif_words)
print('Elapsed Time :', t1-t0, 's')
print()

# Get the K-Most Frequent Words
K = input('DIGIT K TO RETURN THE K-MOST FREQUENT WORDS :')
MFW_1 = wordcount_1.takeOrdered(int(K), key = lambda value: -value[1])
MFW_2 = wordcount_2.takeOrdered(int(K), key = lambda value: -value[1])
MFW_1_r = wordcount_1_r.takeOrdered(int(K), key = lambda value: -value[1])

# Print the Most Frequent Words
print('IMPROVED WORDCOUNT 1 MOST FREQUENT WORDS:')
for w in MFW_1:
    print('-', str(w[0]), ',' , w[1])
print()
print('IMPROVED WORDCOUNT 2 MOST FREQUENT WORDS:')
for w in MFW_2:
    print('-', str(w[0]), ',' , w[1])
print()
print('IMPROVED WORDCOUNT WITH reduceByKey() MOST FREQUENT WORDS:')
for w in MFW_1_r:
    print('-', str(w[0]), ',' , w[1])