#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
from collections import Counter
from nltk.stem import PorterStemmer

def load_stopwords(filepath='stopwords.txt'):
    """Loads stopwords from a file, stripping newline characters."""
    try:
        with open(filepath, 'r') as file:
            # Strip \n and convert to lowercase
            return set(line.strip().lower() for line in file if line.strip())
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Proceeding without stopwords.")
        return set()

def parse_and_preprocess(corpus_path, stop_words):
    """
    Reads the CACM corpus, tokenizes, removes stopwords, and stems terms.
    Returns a dictionary of documents and a unique vocabulary list.
    """
    stemmer = PorterStemmer()
    documents = {}
    vocab = set()
    
    current_doc_id = None
    in_body = False
    doc_terms = []
    
    with open(corpus_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith(".I"):
                # Save previous document before starting a new one
                if current_doc_id is not None:
                    documents[current_doc_id] = doc_terms
                
                current_doc_id = line  # e.g., ".I 1"
                doc_terms = []
                in_body = True
                
            elif line.startswith(".B"):
                in_body = False
                
            elif in_body and not line.startswith("."):
                # Tokenize, lowercase, stopword removal, and stemming
                words = line.split()
                for word in words:
                    word_lower = word.lower()
                    if word_lower not in stop_words:
                        stemmed_word = stemmer.stem(word_lower)
                        doc_terms.append(stemmed_word)
                        vocab.add(stemmed_word)
                        
        # Catch the final document
        if current_doc_id is not None:
            documents[current_doc_id] = doc_terms

    return documents, list(vocab)

def build_tfidf_matrix(documents, vocab):
    """
    Constructs a vectorized TF-IDF matrix for the entire corpus using NumPy.
    """
    num_docs = len(documents)
    vocab_size = len(vocab)
    
    # Map terms to matrix column indices for fast O(1) lookup
    vocab_to_idx = {term: idx for idx, term in enumerate(vocab)}
    doc_ids = list(documents.keys())
    
    # Initialize matrices
    tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float64)
    document_frequency = np.zeros(vocab_size, dtype=np.float64)
    
    print("Building TF-IDF matrix...")
    for i, doc_id in enumerate(doc_ids):
        term_counts = Counter(documents[doc_id])
        
        for term, count in term_counts.items():
            if term in vocab_to_idx:
                idx = vocab_to_idx[term]
                # Log-normalized Term Frequency: 1 + log(tf)
                tf_matrix[i, idx] = 1 + math.log(count)
                document_frequency[idx] += 1

    # Calculate Inverse Document Frequency (IDF)
    # np.maximum prevents division by zero
    idf = np.log(num_docs / np.maximum(document_frequency, 1))
    
    # Vectorized TF-IDF calculation (Broadcast multiplication)
    tfidf_matrix = tf_matrix * idf
    
    # =========================================================================
    # L2 NORMALIZATION (Written from scratch using raw math, actively used)
    # Normally, one uses NumPy's built-in vector norm function
    # norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    # norms[norms == 0] = 1  # Prevent division by zero for empty documents
    # tfidf_matrix = tfidf_matrix / norms
    # For sake of practice, written manually, self-taught L2 normalization
    # =========================================================================
    manual_norms = np.zeros((num_docs, 1), dtype=np.float64)
    for i in range(num_docs):
        sum_squares = 0.0
        for j in range(vocab_size):
            val = tfidf_matrix[i, j]
            sum_squares += val * val
        norm = math.sqrt(sum_squares)
        manual_norms[i, 0] = norm if norm > 0 else 1.0
    tfidf_matrix = tfidf_matrix / manual_norms
    # =========================================================================
    
    return tfidf_matrix, vocab_to_idx, doc_ids

def process_query(query, vocab_to_idx, stop_words):
    """Converts a raw string query into a normalized TF vector."""
    stemmer = PorterStemmer()
    vocab_size = len(vocab_to_idx)
    query_vec = np.zeros(vocab_size, dtype=np.float64)
    
    # Preprocess query exactly like the documents
    words = query.split()
    processed_terms = []
    for word in words:
        word_lower = word.lower()
        if word_lower not in stop_words:
            processed_terms.append(stemmer.stem(word_lower))
            
    # Calculate query TF (1 + log(tf))
    term_counts = Counter(processed_terms)
    for term, count in term_counts.items():
        if term in vocab_to_idx:
            idx = vocab_to_idx[term]
            query_vec[idx] = 1 + math.log(count)
            
    # Normalize query vector using scratch-built L2 normalization
    q_sum_sq = 0.0
    for val in query_vec:
        q_sum_sq += val * val
    q_norm = math.sqrt(q_sum_sq)
    if q_norm > 0:
        query_vec = query_vec / q_norm
        
    return query_vec

def search(query_vec, tfidf_matrix, doc_ids, top_k=5):
    """Computes cosine similarity and returns ranked results."""
    # Since both the matrix and the query vector are normalized, 
    # Cosine Similarity simplifies to the Dot Product
    similarities = tfidf_matrix.dot(query_vec)
    
    # Get indices of the highest scoring documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0:
            results.append((doc_ids[idx], score))
            
    return results

def main():
    corpus_file = 'cacm.all'
    stopwords_file = 'stopwords.txt'
    
    stop_words = load_stopwords(stopwords_file)
    documents, vocab = parse_and_preprocess(corpus_file, stop_words)
    
    if not documents:
        print("Error: No documents loaded. Check 'cacm.all' path.")
        return
        
    tfidf_matrix, vocab_to_idx, doc_ids = build_tfidf_matrix(documents, vocab)
    
    print("\n--- Search Engine Ready ---")
    while True:
        user_input = input("\nEnter search terms (or type 'quit'): ").strip()
        if user_input.lower() == 'quit':
            print("Terminating program...")
            break
        if not user_input:
            continue
            
        query_vec = process_query(user_input, vocab_to_idx, stop_words)
        results = search(query_vec, tfidf_matrix, doc_ids, top_k=10)
        
        if not results:
            print("No matching documents found.")
        else:
            print("\nTop Results:")
            for rank, (doc_id, score) in enumerate(results, 1):
                print(f"{rank}. Document {doc_id} (Similarity: {score:.4f})")

if __name__ == "__main__":
    main()
