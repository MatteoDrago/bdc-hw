
from pyspark import SparkConf, SparkContext
import numpy as np


config = SparkConf().setAppName('Homework 2').setMaster('local')

sc = SparkContext(conf=config)

docs = sc.textFile('text-sample.txt').repartition(8)

N = docs.flatMap(lambda x: x.split()).count() #The total number of words (with repetitions)
print('The total number of word in the dataset, counting repetition, is: ', N)

def howmany(word, x):
    tot=0
    for i in range(len(x)):
        if word == x[i]: tot+=1
    return (word, tot)

from random import randint

partitionSize = round(np.sqrt(N))
print('The dataset has been splitted in ', partitionSize, ' parts')


def partSumBucket(x):
    parS = []
    checked = []
    for word in x: 
        if not word in checked: 
            parS.append( (randint(0, partitionSize-1), howmany(word,x)) )
            checked.append(word)       
    return parS


def secondSum(keyValIter):
    parS = []
    checked = []
    for keyval1 in keyValIter:
        word1 = keyval1[0]
        parSum = 0
        if not word1 in checked:        
            for keyval2 in keyValIter:
                if word1 == keyval2[0]: parSum += keyval2[1]
            parS.append( (word1, parSum) )
            checked.append(word1)      
    return parS

    
words_improved2 = docs.map(lambda doc: (doc.split())).flatMap( partSumBucket ).groupByKey().mapValues(secondSum).flatMap(lambda x: x[1]).reduceByKey(lambda x,y: x+y)

print('The number of different words are: ', words_improved2.count())
