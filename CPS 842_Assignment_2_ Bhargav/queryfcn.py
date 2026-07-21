import os
import math
import sys
import tarfile as tar
import time
import numpy as np

lst = ['who', 'hello', 'is', 'hi']
arr = np.zeros(len(lst))
inp = input("query: ")
lstinpt = inp.split()

for item in lstinpt:
    if item in lst:
        i = lst.index(item)
        arr[i] += 1
    else:
        # Gracefully handle terms not present in the predefined vocabulary list
        print(f"Warning: Term '{item}' is out of vocabulary.")

print(arr)

# ==========================================
#  REUSABLE QUERY VECTORIZATION FUNCTION
# ==========================================

def vectorsim(search_terms, query_string):
    """
    Takes a list of search terms (vocabulary) and a raw query string,
    then constructs and returns a frequency vector mapping query terms to vocabulary indices.
    """
    # Initialize a zero vector matching the length of the vocabulary/search terms list
    query_vector = np.zeros(len(search_terms), dtype=np.float64)
    
    # Tokenize the input query string
    query_tokens = query_string.split()
    
    # Map term frequencies into the vector
    for term in query_tokens:
        if term in search_terms:
            # Find the exact column/index position of the term in our vocabulary list
            index = search_terms.index(term)
            query_vector[index] += 1.0
            
    return query_vector

# Example reuse of the function (replaces the inline script block safely)
# query_input = input("query: ")
# computed_vector = vectorsim(lst, query_input)
# print(computed_vector)
