#!/usr/bin/env python
# coding: utf-8
#Main Search Program, uses file cacm.all, qrels.text, query.text & stopwords.text
# Hemanshu Bhargav 500588413

# For Reference: Below lists the organization of the different parts of the corpus in their respective data structures
# Dictionaries are nested w/ other data structures for fast access
# dicDoc.keys = document id
# dicDoc.values = tuple of terms and term frequency in the document
# wordsDoc.keys = term
# wordsDoc.values = document frequency (documents the term occurs in)

#import os
import math
#import sys
import tarfile as tar
#import time
import numpy as np
from sklearn.metrics import average_precision_score
from nltk.stem import PorterStemmer
#from nltk.stem.porter import *

# Extract the file
# fp = tar.open("cacm.tar.gz")
# file_extractor = fp.extractall()
cacm = open('cacm.all', 'r')
q1 = open('qrels.text', 'r')
q2 = open('query.text', 'r')

def invert(cacm):
    # Create an empty dictionary (implemented as hashmap in python)
    dicDoc = {}
    key = ""
    writeInFile = False
    docBody = []
    # automatically closes after opening
    with open(cacm, 'r') as file:
        for line in file.readlines():
            # document ID '.I', title '.T', and abstract '.W' all occur before '.B'
            if ".B" in line:
                writeInFile = False
                # Store term frequencies
                dictFreq = {}
                item: str
                for item in docBody:
                    dictFreq[item] = dictFreq.get(item, 0) + 1
                # enumerate the body (Terms) and key is Document ID '.I'
                dicDoc[key] = [list(enumerate(docBody)), dictFreq]
                docBody = []

            elif writeInFile:
                if len(line.strip()) != 0:
                    # split the terms from the lines into seperate lines
                    docBody.extend(line.strip().split())

            elif ".I" in line:
                key = line.strip()
                writeInFile = True

    # For postings file, extract terms and their document id from dictionary
    # wordsdoc = terms and the documents they occur in
    wordsDoc = {}
    for doc in dicDoc:
        for key in dicDoc[doc][1].keys():
            # if not already present
            if key not in wordsDoc.keys():
                wordsDoc[key] = [doc]
            else:
                # key = term, value is all documents containing that key
                wordsDoc[key] = wordsDoc[key] + [doc]

    # Above code continued and terms sorted alphabetically
    search = sorted(wordsDoc.keys())
    #      print(key, " - is in documents: ", wordsDoc[key])

    # wordsDoc =  terms and the documents they occur in
    search_terms = list(search)

    def stopwords(search_terms):
        stop = open('stopwords.txt', 'r')
        criteria = set(stop)
        # below 2 line snippet retrieved from https://pythonprogramming.net/stop-words-nltk-tutorial/ on October 4, 2019
        filtered_words = [w for w in search_terms if w not in criteria]
        return filtered_words

    def mystemmer(stopped_words):
        stemming = PorterStemmer()
        token = str(stopped_words) #FIX THIS?? causes search_terms to be one string of corpus
        #for this reason search_Terms was used
        stemmed_terms = stemming.stem(token)
        return stemmed_terms

    stopped_words = stopwords(search_terms)
    search_terms = mystemmer(stopped_words)

    return dicDoc, wordsDoc, search_terms

cacm = 'cacm.all'
dicDoc, wordsDoc, search_terms = invert(cacm)


# =============================================================================
# def testv2(dicDoc, wordsDoc, search_terms):
#     queryin = ""
#     while (1):
#         queryin = input("Enter the term that you want to search: ")
#         if queryin == "quit":
#             print("You've terminated the program!")
#             break
#         else:
#             print(queryin, " - is in documents: ", wordsDoc.get(queryin.strip(),"NONE, Does not Exist"))
#             #print("Document ", dicDoc.keys(), "contains the following frequencies of each term", dicDoc.values())
#     return queryin
# =============================================================================

# Compute term frequencies
search_Terms = set({}) #recall that sets are unordered, converted to list below. Here, creates a
#set of distinct terms (Removal of duplicates before actual frequency math happens)
for docKey in dicDoc.keys():
    search_Terms = search_Terms.union(dicDoc[docKey][1].keys())
#convert to list becuase list is an ordered data structure, won't lose order from here on out
search_Terms = list(search_Terms)

docTFvectors = {}
DF = np.zeros(len(search_Terms),dtype=np.float64)
for docKey in dicDoc.keys():
    tf = np.zeros(len(search_Terms),dtype=np.float64)
    index = 0
    #tf = actual occurence of term/doc, TF = weighted tf using log
    for term in search_Terms:                
        if dicDoc[docKey][1].get(term,0) != 0:            
            tf[index] = 1+ math.log(dicDoc[docKey][1].get(term,0))
            DF[index] += dicDoc[docKey][1].get(term,0)
        index += 1
    docTFvectors[docKey] = tf

IDF = np.zeros(len(search_Terms),dtype=np.float64)  
i = 0 
for term in search_Terms:
    IDF[i] = math.log(3204/DF[i])
    i += 1

docsTFIDFvectors={}
for aDoc in docTFvectors.keys():
    tfidf = np.zeros(len(search_Terms),dtype=np.float64)
    i = 0
    normalizationDeno = 0
    for freq in docTFvectors[aDoc]:
        tfidf[i] = docTFvectors[aDoc][i]*IDF[i]
        normalizationDeno +=  tfidf[i] * tfidf[i]
        i+=1
    sqrtND = normalizationDeno **.5
    finaldocvector = tfidf
    if sqrtND > 0:
        finaldocvector = tfidf/sqrtND
    #docsTFIDFvectors[aDoc] = vectors of TF, where aDoc = docID
    docsTFIDFvectors[aDoc] = finaldocvector
    # For testing: prints all normalized term frequencies > 0
#    for i in range(len(docsTFIDFvectors.values())): 
 #       print(list(docsTFIDFvectors.values())[i].nonzero())  

## Run Search Program: User intereaction, query and doc cosine similarity, 
#    # all the action happens here. 
#    
def userinterface(search_Terms, docsTFIDFvectors):
    #initialize a vector of 0s for the query whose length is equal to length of the 
    #searhable corpus
     queryvec = np.zeros(len(search_Terms), dtype=np.float64)
     # infinite loop prompted user for queries (unless user types quit)
     while(1):
         inp = input("Enter the terms that you wish to search: ")
         if inp == "quit":
             print("You've terminated the program!")
             break
         else:
             #take all arguments and tokenize using .split()
             lstinpt = inp.split()
             #for each token in "argv", repeat normalization procedure
             for item in lstinpt:
                 i = search_Terms.index(item)
                 queryvec[i] += 1 #if query term is in searchable terms, add to query vector
#                #print(list(queryvec.nonzero()))
                 finalqueryTFvectors = {}
                 finalqueryTFvectors = querytermfreq(queryvec, docsTFIDFvectors, search_Terms)
                 #myqueryTFvectors = dict whose values are term frequencies of each item in corpus
                 #found in query
                 #Now compute cosine similarity
                 denominator_for_docs = denomOfcosineSim(docsTFIDFvectors)
               #  denominator_for_queries = denomOfcosineSim(finalqueryTFvectors)
               #  cosineSim(docsTFIDFvectors, denominator_for_docs,denominator_for_queries, finalqueryTFvectors)
                 
                 
 #NOTE: BAD PRACTICE. Both normalization & term freq should use modular shared components. 
 #For now, copy-pasted & modified
def querytermfreq(queryvec, docsTFIDFvectors, search_Terms):
    #Side Note on data structures: queryvec's index isn't docIDS like docsTFIDFvectors because array vs dict
    # Now, to obtain term frequencies of terms in query, use 1+log(term frequency) weight scheme
    #(IDF is not considered)
    #see if word is found in corpus (each item in search_Terms) as done for documents

    queryTFvectors = {} #create a dict for query term freq, just like doc term freq, with index as docID
    #query vec replaces DF
    for docKey in dicDoc.keys():
        tf = np.zeros(len(search_Terms),dtype=np.float64)
        qindex = 0
        #tf = actual occurence of term/doc, TF = weighted tf using log
        for term in search_Terms:                
            if dicDoc[docKey][1].get(term,0) != 0:            
                tf[qindex] = 1+ math.log(dicDoc[docKey][1].get(term,0)) #1+log(term frequency)
                queryvec[qindex] += dicDoc[docKey][1].get(term,0)
            qindex += 1
        queryTFvectors[docKey] = tf
        return queryTFvectors
'''
Explanation of cosine similarity logic
remember, finalqueryTFvectors keys comprise the same 3204 docs
Query Normalization is omitted. See reasoning below:
because IDF is omitted, and 1+log(f) or 1+log(term freq in query) is used instead,
all term frequencies of query are either 1 or 0
Using this method, normalization of queries is not required (but normally it is)
here we used a special variation of the normal 1+lof(f)*IDF query tf formula
 numerator is docsTFIDFvectors.values * myqueryTFvectorsvalues
 denominator is all doc term frequencies squared, and the sq root of the sum of all squares 
 then multiplied by all query term frequencies squared and sq root of sum of all sqs
 but again, since values of query term freq are 1 or 0, simply take sum of all query term frequencies
'''
def denomOfcosineSim(dict):
    #Square docsTFIDFvectors.values, sum and take square root (for denominator)
    j = 0 #initialize j to perserve index post-loop -->MAY BE unnecessary following last code edit
    for j in range(len(dict.values())): 
        dociter = list(dict.values())  
        if (j==0):
            accumulator = dociter[0] # set a running total beginning with first frequency
        dociter[j] = dociter[j]*dociter[j] 
        if (j!=0):
            accumulator += dociter[j]  #sum the current and previous frequency
        sqrootOfallSums = accumulator**.5
     #NUMERATOR   cos_weights = list(sqrootOfallSums) #store all document/query vectors w/ their weights into list
#    print(sqrootOfallSums)
    return sqrootOfallSums #cos_weights

def cosineSim(docsTFIDFvectors, denominator_for_docs,denominator_for_queries, finalqueryTFvectors):
    numerator = 0 #loop through term freqencies in doc & query and multiply
    for i in docsTFIDFvectors.values:
        docweights = np.array(docsTFIDFvectors.values)
    for j in finalqueryTFvectors.values:
        queryweights = np.array(finalqueryTFvectors.values)
    for weight in docweights:
        for compareweight in queryweights:
            numerator = np.dot(docweights, queryweights)
    denominator = denominator_for_docs * denominator_for_queries 
    similarity_score = numerator/denominator
    # Now need to sort documents based on similarity scores for ranked retrieval
    # ....
userinterface(search_Terms, docsTFIDFvectors)               
                 

#             # Now we perform cosine similarity b/w the query vectors & all 3204 doc vectors
#             for j in docsTFIDFvectors:
#                 doctotal = 0
#                 doctotal += docsTFIDFvectors[j]
#                 for k in queryvec:
#                     querytotal = np.zeros(len(docsTFIDFvectors),dtype=np.float)
#                     querytotal += queryvec[k]
#                 cos_sim = np.zeros(len(docsTFIDFvectors),dtype=np.float)
#                 cos_sim = (finaldocvector[j] * queryvec[k])/(doctotal/querytotal)
#             for docscore in docsTFIDFvectors:#Filter based on top-k retrieval of IDF > 0.5
#                 for score in cos_sim:
#                     if docsTFIDFvectors[docscore] > 0.5:
#                         print(sorted(docsTFIDFvectors[docscore]))
#             return cos_sim
#             #print(queryvec)
#
#def eval(q1,q2,cos_sim):
#    for queries in q2:
#        queries = list(q2.split())
#    for qrels in q1:
#        qrels = list(q1.split())
#        true_Scores = np.array(qrels)
#        average_precision_score(true_Scores, cos_sim)  
#    
