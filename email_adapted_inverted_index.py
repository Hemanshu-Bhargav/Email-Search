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
                key = line.strip().split()[-1] # Extract just the document ID number
                writeInFile = True

    # For postings file, extract terms and their document id from dictionary
    # wordsdoc = terms and the documents they occur in
    wordsDoc = {}
    for doc in dicDoc:
        for term in dicDoc[doc][1].keys():
            # if not already present
            if term not in wordsDoc.keys():
                wordsDoc[term] = [doc]
            else:
                # key = term, value is all documents containing that key
                wordsDoc[term] = wordsDoc[term] + [doc]

    # Above code continued and terms sorted alphabetically
    search = sorted(wordsDoc.keys())
    search_terms = list(search)

    def stopwords(search_terms):
        if os.path.exists('stopwords.txt'):
            with open('stopwords.txt', 'r') as stop:
                criteria = set(line.strip().lower() for line in stop)
        else:
            criteria = {'the', 'is', 'at', 'which', 'and', 'a', 'an', 'in', 'to', 'of', 'for', 'on', 'with'}
        # below 2 line snippet retrieved from https://pythonprogramming.net/stop-words-nltk-tutorial/ on October 4, 2019
        filtered_words = [w for w in search_terms if w not in criteria]
        return filtered_words

    def mystemmer(stopped_words):
        stemming = PorterStemmer()
        stemmed_terms = [stemming.stem(str(token)) for token in stopped_words]
        return stemmed_terms

    stopped_words = stopwords(search_terms)
    search_terms = mystemmer(stopped_words)

    return dicDoc, wordsDoc, search_terms

# Build inverted index
dicDoc, wordsDoc, search_terms = invert(cacm)

def parse_queries(query_file):
    """Parses query.txt into a dictionary mapping query ID to query text string."""
    queries = {}
    current_qid = None
    current_text = []
    
    for line in query_file:
        line_str = line.strip()
        if line_str.startswith(".I"):
            if current_qid and current_text:
                queries[current_qid] = " ".join(current_text)
                current_text = []
            parts = line_str.split()
            if len(parts) > 1:
                current_qid = parts[1]
        elif line_str.startswith(".") and len(line_str) == 2:
            continue
        elif current_qid:
            if line_str:
                current_text.append(line_str)
    if current_qid and current_text:
        queries[current_qid] = " ".join(current_text)
        
    # Fallback parser for standard line-by-line query files
    if not queries:
        query_file.seek(0)
        for line in query_file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries

def parse_qrels(qrels_file):
    """Parses qrels.txt into a dictionary mapping query ID to a set of relevant document IDs."""
    qrels = {}
    for line in qrels_file:
        parts = line.strip().split()
        if len(parts) >= 4:
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
        elif len(parts) == 2:
            qid, docid, rel = parts[0], parts[1], 1
        else:
            continue
        
        if rel > 0:
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(docid)
    return qrels

'''
@param dicDoc = dictionary of archived documents constructed in function invert
@param wordsDoc = dictionary of terms mapped to lists of document IDs
@param queries = dictionary of query ID mapped to query text
'''
def search(dicDoc, wordsDoc, queries):
    """Performs retrieval using a TF-IDF vector space model from scratch."""
    results = {}
    N = len(dicDoc)
    stemmer = PorterStemmer()
    
    # Precompute Inverse Document Frequency (IDF) for all terms
    idf = {}
    for term, docs in wordsDoc.items():
        df = len(docs)
        idf[term] = math.log((N / df) + 1)
        
    for qid, qtext in queries.items():
        query_terms = [stemmer.stem(w.lower()) for w in qtext.split()]
        q_tf = {}
        for qt in query_terms:
            q_tf[qt] = q_tf.get(qt, 0) + 1
            
        doc_scores = {}
        for qt, qfreq in q_tf.items():
            if qt in wordsDoc:
                matching_docs = wordsDoc[qt]
                term_idf = idf.get(qt, 1.0)
                for doc in matching_docs:
                    doc_tf = dicDoc[doc][1].get(qt, 0)
                    if doc_tf > 0:
                        score = doc_tf * term_idf * qfreq * term_idf
                        doc_scores[doc] = doc_scores.get(doc, 0.0) + score
                        
        # Sort documents by relevance score descending
        sorted_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        results[qid] = sorted_docs
        
    return results

# Prepare query and relevance files
q1.seek(0)
q2.seek(0)
qrels_data = parse_qrels(q1)
queries_data = parse_queries(q2)

# Execute search against all queries
retrieved_results = search(dicDoc, wordsDoc, queries_data)

'''
@param retrieved_results = dictionary of query IDs mapped to ranked lists of retrieved document IDs
@param qrels = dictionary of query IDs mapped to sets of true relevant document IDs
'''
def eval(retrieved_results, qrels):
    total_ap = 0.0
    total_r_prec = 0.0
    num_queries = 0
    
    print("\n" + "=" * 60)
    print(f"{'Query ID':<12}{'Average Precision':<22}{'R-Precision':<15}")
    print("=" * 60)
    
    for qid, rel_docs in qrels.items():
        if not rel_docs:
            continue
        
        retrieved = retrieved_results.get(qid, [])
        R = len(rel_docs)
        num_queries += 1
        
        relevant_count = 0
        precision_sum = 0.0
        
        # Calculate Average Precision (AP) from scratch
        for i, doc in enumerate(retrieved):
            rank = i + 1
            if doc in rel_docs:
                relevant_count += 1
                precision_at_k = relevant_count / rank
                precision_sum += precision_at_k
                
        ap = precision_sum / R if R > 0 else 0.0
        total_ap += ap
        
        # Calculate R-Precision from scratch (precision at rank R)
        top_R_retrieved = retrieved[:R]
        r_relevant_count = sum(1 for doc in top_R_retrieved if doc in rel_docs)
        r_prec = r_relevant_count / R if R > 0 else 0.0
        total_r_prec += r_prec
        
        print(f"{qid:<12}{ap:<22.4f}{r_prec:<15.4f}")
        
    # Calculate Mean Average Precision (MAP) and Mean R-Precision
    map_score = total_ap / num_queries if num_queries > 0 else 0.0
    mean_r_prec = total_r_prec / num_queries if num_queries > 0 else 0.0
    
    print("=" * 60)
    print(f"MAP is: {map_score:.4f}")
    print(f"R-precision is: {mean_r_prec:.4f}")
    print("=" * 60)

# Run Evaluation Program
eval(retrieved_results, qrels_data)
