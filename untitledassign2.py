import os
import math
import sys
import tarfile as tar
import time
from nltk.stem import PorterStemmer

cacm = open('cacm.all', 'r')

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
    filtered_words = [w for w in search_terms if w not in criteria]
    return filtered_words

def mystemmer(stopped_words):
    stemming = PorterStemmer()
    token = str(stopped_words)
    stemmed_terms = stemming.stem(token)
    return stemmed_terms

stopped_words = stopwords(search_terms)
search_terms = mystemmer(stopped_words)

for term in wordsDoc.keys():
   # corpus = set(term) -->creates set of characters
    print(term)

