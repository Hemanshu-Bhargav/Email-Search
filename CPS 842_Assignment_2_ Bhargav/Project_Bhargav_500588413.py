#!/usr/bin/env python
# coding: utf-8
# Main Search Program, uses file cacm.all, qrels.text, query.text & stopwords.text
# Hemanshu Bhargav 500588413

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
import numpy as np
from sklearn.metrics import average_precision_score
from nltk.stem import PorterStemmer
#from nltk.stem.porter import *


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
                # Extract just the document ID string cleanly
                key = line.strip().split()[-1] if len(line.strip().split()) > 1 else line.strip()
                writeInFile = True

    # For postings file, extract terms and their document id from dictionary
    # wordsdoc = terms and the documents they occur in
    wordsDoc = {}
    for doc in dicDoc:
        for term_key in dicDoc[doc][1].keys():
            # if not already present
            if term_key not in wordsDoc.keys():
                wordsDoc[term_key] = [doc]
            else:
                # key = term, value is all documents containing that key
                wordsDoc[term_key] = wordsDoc[term_key] + [doc]

    # Above code continued and terms sorted alphabetically
    search = sorted(wordsDoc.keys())
    #      print(key, " - is in documents: ", wordsDoc[key])

    # wordsDoc = terms and the documents they occur in
    search_terms = list(search)

    def stopwords(terms_list):
        if os.path.exists('stopwords.txt'):
            with open('stopwords.txt', 'r', encoding='utf-8') as stop:
                criteria = set(line.strip().lower() for line in stop)
        else:
            criteria = {'the', 'is', 'at', 'which', 'and', 'a', 'an', 'in', 'to', 'of', 'for', 'on', 'with'}
        # below 2 line snippet retrieved from https://pythonprogramming.net/stop-words-nltk-tutorial/ on October 4, 2019
        filtered_words = [w for w in terms_list if w.lower() not in criteria]
        return filtered_words

    def mystemmer(stopped_words):
        stemming = PorterStemmer()
        # FIX THIS?? Handled type issue by iterating over list tokens rather than passing the list as a string
        stemmed_terms = [stemming.stem(str(token)) for token in stopped_words]
        return stemmed_terms

    stopped_words = stopwords(search_terms)
    search_terms = mystemmer(stopped_words)

    return dicDoc, wordsDoc, search_terms

cacm = 'cacm.all'
dicDoc, wordsDoc, search_terms = invert(cacm)
