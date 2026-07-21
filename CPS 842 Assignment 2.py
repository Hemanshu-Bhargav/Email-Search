#!/usr/bin/env python
# coding: utf-8

# For Reference: Below lists the organization of the different parts of the corpus in their respective data structures
# Dictionaries are nested w/ other data structures for fast access
# dicDoc.keys = document id
# dicDoc.values = tuple of terms and term frequency in the document
# wordsDoc.keys = term
# wordsDoc.values = document frequency (documents the term occurs in)

import os
import math
import sys
import tarfile as tar
import time
from nltk.stem import PorterStemmer

# Extract the file
# fp = tar.open("cacm.tar.gz")
# file_extractor = fp.extractall()
cacm = open('cacm.all', 'r')
q1 = open('qrels.txt', 'r')
q2 = open('query.txt', 'r')

'''
@param corpusfile = file with data seperated by the new line operator
'''
def invert(corpusfile):
    # Create an empty dictionary (implemented as hashmap in python)
    dicDoc = {}
    key = ""
    writeInFile = False
    docBody = []
    # automatically closes after opening
    with open('cacm.all', 'r') as file:
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
        token = str(stopped_words)
        stemmed_terms = stemming.stem(token)
        return stemmed_terms

    stopped_words = stopwords(search_terms)
    search_terms = mystemmer(stopped_words)

    return dicDoc, wordsDoc, search_terms

#Build inverted index
dicDoc, wordsDoc, search_terms = invert(cacm)

def testv2(dicDoc, wordsDoc, search_terms):
    while (1):
        queryin = input("Enter the term that you want to search: ")
        if queryin == "ZZEND":
            print("You've terminated the program!")
            break
        else:
            print(queryin, " - is in documents: ", wordsDoc[key])
            #print("Document ", dicDoc.keys(), "contains the following frequencies of each term", dicDoc.values())
    return queryin


# Run Search Program (This line marks the first interaction with the the user)
# User is prompted to enter a query and the index function, invert, is run
testv2(dicDoc, wordsDoc, search_terms)

'''
@param dicDoc = dictionary of archived documents constructed in function invert
@param wordsDoc = dictionary of archived documents constructed in function invert
@param queryin = dictionary of archived documents constructed in function invert
'''
def search(dicDoc, wordsDoc, queryin):
    # set provides us with a distinct (all duplicates removed) set of all terms (from wordsDoc.keys)
    # list then sorts the set and the resultant data structure contains all 10,000 terms of the corpus in an ordered format

    #corpus_list = list(set(dicDoc.values()))  ------->sets are not hashable, ignore for now, can fix during project
    vectordict = {} #Create a dictionary to hold term weights and IDF of each term
    corpus_list = list(dicDoc.values())

    #print(dicDoc.keys())
    vectordoc = []
    vector = [vectordoc,[]]
    for words in corpus_list:
        print("Loop Test")
        for keys in dicDoc.keys():
            vectordoc = keys
        for lists in dicDoc.values():
            vector[1][0] = dicDoc.values[0]
            vector[1][1] = dicDoc.values[0] #TF of term in document
        #Each term has its own IDF
        for key in vectordict:
            TF = int(vector[1][1])
            IDF = 1 + math.log(TF)
            vectordict.keys = IDF
    #Weight Calculation for each term, remember, 3204 documents, and each document has 10,000 terms/weights roughly
    #LOST COPY AFTER SUBMISSION. COMPLETE CALCULATIONS.
        for term, TF in dicDoc.items():
            weight = vectordict.keys*TF

        # Need to check each term in the corpus_list against all 3204 documents in the CACM collection
        # To do so, we will compare each word against each document for the word's existence in that document
        # DicDoc's key-value pairs provide us the documents, each key is a document ID, and its value is the document
        # Set each document ID (3204 total) to a vector
    # dicDoc.keys = docIDs, and dicDoc.values already has terms ordered by docID and TF per document

#Construct inverted index
search(dicDoc, wordsDoc)
#To Do: Add evaluation from
def eval(vectors, q1, q2):
    print("MAP is: ")
    print("R-precision is: ")
