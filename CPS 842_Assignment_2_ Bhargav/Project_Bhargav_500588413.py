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

import math
import numpy as np
from nltk.stem import PorterStemmer

def load_stopwords():
    stop = open('stopwords.txt', 'r')
    # Fixed: strip newline characters so stopword filtering actually works
    criteria = set(line.strip().lower() for line in stop if line.strip())
    stop.close()
    return criteria

def invert(cacm):
    stop_words = load_stopwords()
    stemming = PorterStemmer()
    
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
                for item in docBody:
                    dictFreq[item] = dictFreq.get(item, 0) + 1
                # enumerate the body (Terms) and key is Document ID '.I'
                dicDoc[key] = [list(enumerate(docBody)), dictFreq]
                docBody = []

            elif writeInFile:
                if len(line.strip()) != 0:
                    # split the terms from the lines into separate words, lowercase, and stem individually
                    for word in line.strip().split():
                        word_lower = word.lower()
                        if word_lower not in stop_words:
                            stemmed_term = stemming.stem(word_lower)
                            docBody.append(stemmed_term)

            elif ".I" in line:
                key = line.strip()
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
    # wordsDoc = terms and the documents they occur in
    search_terms = list(search)

    return dicDoc, wordsDoc, search_terms

cacm = 'cacm.all'
dicDoc, wordsDoc, search_terms = invert(cacm)

# Compute term frequencies
search_Terms = set({}) # recall that sets are unordered, converted to list below. Here, creates a
# set of distinct terms (Removal of duplicates before actual frequency math happens)
for docKey in dicDoc.keys():
    search_Terms = search_Terms.union(dicDoc[docKey][1].keys())

# convert to list because list is an ordered data structure, won't lose order from here on out
search_Terms = list(search_Terms)

docTFvectors = {}
DF = np.zeros(len(search_Terms), dtype=np.float64)

for docKey in dicDoc.keys():
    tf = np.zeros(len(search_Terms), dtype=np.float64)
    index = 0
    # tf = actual occurrence of term/doc, TF = weighted tf using log
    for term in search_Terms:            
        if dicDoc[docKey][1].get(term, 0) != 0:            
            tf[index] = 1 + math.log(dicDoc[docKey][1].get(term, 0))
            # Fixed: DF tracks Document Frequency (how many docs contain the term), increment by 1 per doc
            DF[index] += 1 
        index += 1
    docTFvectors[docKey] = tf

IDF = np.zeros(len(search_Terms), dtype=np.float64)  
i = 0 
total_docs = len(dicDoc)
for term in search_Terms:
    if DF[i] > 0:
        IDF[i] = math.log(total_docs / DF[i])
    else:
        IDF[i] = 0.0
    i += 1

docsTFIDFvectors = {}
for aDoc in docTFvectors.keys():
    tfidf = np.zeros(len(search_Terms), dtype=np.float64)
    i = 0
    normalizationDeno = 0
    for freq in docTFvectors[aDoc]:
        tfidf[i] = docTFvectors[aDoc][i] * IDF[i]
        normalizationDeno += tfidf[i] * tfidf[i]
        i += 1
    sqrtND = normalizationDeno ** .5
    finaldocvector = tfidf
    if sqrtND > 0:
        finaldocvector = tfidf / sqrtND
    # docsTFIDFvectors[aDoc] = vectors of TF, where aDoc = docID
    docsTFIDFvectors[aDoc] = finaldocvector

## Run Search Program: User interaction, query and doc cosine similarity, 
# all the action happens here.    
def userinterface(search_Terms, docsTFIDFvectors):
    # initialize a vector of 0s for the query whose length is equal to length of the searchable corpus
    while(1):
        inp = input("Enter the terms that you wish to search: ")
        if inp == "quit":
            print("You've terminated the program!")
            break
        if not inp.strip():
            continue
            
        queryvec = np.zeros(len(search_Terms), dtype=np.float64)
        # take all arguments and tokenize using .split()
        lstinpt = inp.split()
        stemming = PorterStemmer()
        stop_words = load_stopwords()
        
        # for each token in query, preprocess and match to search_Terms index
        for item in lstinpt:
            item_lower = item.lower()
            if item_lower not in stop_words:
                stemmed_item = stemming.stem(item_lower)
                if stemmed_item in search_Terms:
                    i = search_Terms.index(stemmed_item)
                    queryvec[i] += 1 # if query term is in searchable terms, add to query vector

        # Compute query term frequencies (1 + log(tf))
        finalqueryTFvectors = np.zeros(len(search_Terms), dtype=np.float64)
        for i in range(len(search_Terms)):
            if queryvec[i] > 0:
                finalqueryTFvectors[i] = 1 + math.log(queryvec[i])

        # Compute cosine similarity across all document vectors
        scores = {}
        for docKey in docsTFIDFvectors.keys():
            docweights = docsTFIDFvectors[docKey]
            # Numerator: dot product of doc vector and query vector
            numerator = np.dot(docweights, finalqueryTFvectors)
            
            # Denominator: product of L2 norms
            doc_norm = np.linalg.norm(docweights)
            query_norm = np.linalg.norm(finalqueryTFvectors)
            denominator = doc_norm * query_norm
            
            if denominator > 0:
                similarity_score = numerator / denominator
                if similarity_score > 0:
                    scores[docKey] = similarity_score

        # Sort documents based on similarity scores for ranked retrieval
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_results:
            print("No matching documents found.")
        else:
            print("\nTop Results:")
            for rank, (doc_id, score) in enumerate(sorted_results[:10], 1):
                print(f"{rank}. Document {doc_id} (Score: {score:.4f})")

if __name__ == "__main__":
    userinterface(search_Terms, docsTFIDFvectors)
