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
from nltk.stem import PorterStemmer

def load_stopwords(filepath='stopwords.txt'):
    """Loads stopwords manually into a set, trimming newlines."""
    stop_words = set()
    try:
        with open(filepath, 'r') as file:
            for line in file:
                cleaned_line = line.strip().lower()
                if cleaned_line:
                    stop_words.add(cleaned_line)
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Continuing without stopwords.")
    return stop_words

def parse_corpus(cacm_path, stop_words, stemmer):
    """
    Parses cacm.all from scratch, builds term frequencies per document,
    and returns document dictionary and vocabulary.
    """
    doc_data = {}
    current_doc_id = None
    in_body = False
    tokens = []

    with open(cacm_path, 'r') as file:
        for line in file:
            line_str = line.strip()

            if line_str.startswith(".I"):
                if current_doc_id is not None:
                    tf_dict = {}
                    for term in tokens:
                        tf_dict[term] = tf_dict.get(term, 0) + 1
                    doc_data[current_doc_id] = [list(enumerate(tokens)), tf_dict]

                current_doc_id = line_str
                tokens = []
                in_body = True

            elif line_str.startswith(".B"):
                in_body = False

            elif in_body and not line_str.startswith("."):
                words = line_str.split()
                for word in words:
                    word_lower = word.lower()
                    if word_lower not in stop_words:
                        stemmed_word = stemmer.stem(word_lower)
                        tokens.append(stemmed_word)

        if current_doc_id is not None:
            tf_dict = {}
            for term in tokens:
                tf_dict[term] = tf_dict.get(term, 0) + 1
            doc_data[current_doc_id] = [list(enumerate(tokens)), tf_dict]

    return doc_data

def build_inverted_index(dicDoc):
    """Builds wordsDoc mapping terms to document IDs containing them."""
    wordsDoc = {}
    for doc in dicDoc:
        for key in dicDoc[doc][1].keys():
            if key not in wordsDoc:
                wordsDoc[key] = [doc]
            else:
                wordsDoc[key].append(doc)
    return wordsDoc

def compute_tfidf_and_normalize(dicDoc):
    """
    Computes TF-IDF and Euclidean Normalization (L2 norm) 
    completely manually using explicit loops and basic math.
    """
    total_docs = len(dicDoc)

    # 1. Compute Document Frequency (DF) for each term manually
    df_dict = {}
    for docKey in dicDoc.keys():
        for term in dicDoc[docKey][1].keys():
            df_dict[term] = df_dict.get(term, 0) + 1

    # 2. Compute Inverse Document Frequency (IDF) manually
    idf_dict = {}
    for term, df in df_dict.items():
        idf_dict[term] = math.log(total_docs / df)

    # 3. Compute TF-IDF weights and Euclidean Norm per document
    normalized_doc_vectors = {}
    for docKey, doc_content in dicDoc.items():
        sum_of_squares = 0.0
        doc_weights = {}

        for term, count in doc_content[1].items():
            tf_weight = 1.0 + math.log(count)
            tfidf = tf_weight * idf_dict[term]
            doc_weights[term] = tfidf
            sum_of_squares += tfidf * tfidf

        norm_denominator = math.sqrt(sum_of_squares)

        norm_vector = {}
        if norm_denominator > 0:
            for term, weight in doc_weights.items():
                norm_vector[term] = weight / norm_denominator

        normalized_doc_vectors[docKey] = norm_vector

    return normalized_doc_vectors, idf_dict

def process_query(query_str, stop_words, stemmer, idf_dict):
    """Preprocesses user query and computes normalized query vector manually."""
    words = query_str.split()
    query_tokens = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower not in stop_words:
            query_tokens.append(stemmer.stem(word_lower))

    query_tf = {}
    for term in query_tokens:
        query_tf[term] = query_tf.get(term, 0) + 1

    sum_of_squares = 0.0
    query_weights = {}
    for term, count in query_tf.items():
        if term in idf_dict:
            tf_weight = 1.0 + math.log(count)
            query_weights[term] = tf_weight
            sum_of_squares += tf_weight * tf_weight

    query_norm = math.sqrt(sum_of_squares)

    normalized_query = {}
    if query_norm > 0:
        for term, weight in query_weights.items():
            normalized_query[term] = weight / query_norm

    return normalized_query

def calculate_cosine_similarity(norm_doc_vec, norm_query_vec):
    """Calculates dot product of two normalized sparse vectors manually."""
    dot_product = 0.0
    for term, q_weight in norm_query_vec.items():
        if term in norm_doc_vec:
            dot_product += norm_doc_vec[term] * q_weight
    return dot_product

def main():
    cacm_path = 'cacm.all'
    stopwords_path = 'stopwords.txt'

    stemmer = PorterStemmer()
    stop_words = load_stopwords(stopwords_path)

    print("Parsing corpus...")
    dicDoc = parse_corpus(cacm_path, stop_words, stemmer)
    if not dicDoc:
        print("Error: Could not read corpus.")
        return

    wordsDoc = build_inverted_index(dicDoc)
    normalized_doc_vectors, idf_dict = compute_tfidf_and_normalize(dicDoc)

    print("\n--- Search Engine Ready ---")
    while True:
        user_input = input("\nEnter search terms (or type 'quit'): ").strip()
        if user_input.lower() == 'quit':
            print("Terminating program...")
            break
        if not user_input:
            continue

        query_vector = process_query(user_input, stop_words, stemmer, idf_dict)
        
        results = []
        for doc_id, doc_vector in normalized_doc_vectors.items():
            score = calculate_cosine_similarity(doc_vector, query_vector)
            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda item: item[1], reverse=True)

        if not results:
            print("No matching documents found.")
        else:
            print("\nTop 10 Results:")
            for rank, (doc_id, score) in enumerate(results[:10], 1):
                print(f"{rank}. Document {doc_id} (Similarity: {score:.4f})")

if __name__ == "__main__":
    main()
