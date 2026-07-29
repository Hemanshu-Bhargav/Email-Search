#!/usr/bin/env python
# coding: utf-8
# ==========================================
# PREFACE
''' First, extract a user's gmail for email messages which will form the corpus
This corpus will then be exported to Google Sheets.
As explained in the proposal, the intended usage of this program
is to "bridge the gap" that exists for Google Add-on development.
Although, search within Google exists, if any GMail account users wishes
to install an add-on for increased functionality, the onus is on the developer
of that add-on to implement a reliable search program which is compatible
with Google's services. 
This program aims to be a portable solution.
Note: Understandably, python's execution is not as fast, as say Java's,
because of inherent differences in execution of compiled and interpreted languages.
Nonetheless, Python was chosen for portability being cognizant of vast library support Python offers for NLP tasks.
Add-ons can only use Google sheets for their database and because parsing attachments
is both out of the scope of this project and ethically questionable, therefore attachments are ignored. '''
# ==========================================
# ==========================================
# UNIFIED EMAIL SEARCH & VECTOR SPACE ENGINE
# Combines IMAP/Google Sheets integration with
# CPS 842 TF-IDF and Cosine Similarity Retrieval
# ==========================================

import email
import imaplib
import json
import os
import re
import getpass
import math
import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem import PorterStemmer
import pygsheets

# ==========================================
# CONFIGURATION
# ==========================================
GOOGLE_CREDENTIALS_FILE = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "path/to/your/service_account.json"
)
GOOGLE_SHEET_NAME = "CPS 842 Project V1"

# ==========================================
# IMAP AUTHENTICATION & EMAIL FETCHING
# ==========================================

def user_prompt():
    """Prompts user for Gmail credentials and recipient target."""
    username = input("Please enter your Gmail username: ")
    password = getpass.getpass("Enter your password: ")
    recipient = input("Please enter whose emails you'd like to store (e.g., sender@gmail.com): ")
    return username, password, recipient

def fetch_emails(username, password, recipient):
    """Connects to Gmail via IMAP and extracts emails from a specific sender."""
    try:
        sign_in_link = imaplib.IMAP4_SSL("imap.gmail.com")
        sign_in_link.login(username, password)
        sign_in_link.select("INBOX")
        
        resp, items = sign_in_link.search(None, 'FROM', recipient)
        if resp != 'OK' or not items[0]:
            print("No messages found or failed to search inbox.")
            return {}

        all_emails = {}
        for emailid in items[0].split():
            resp, data = sign_in_link.fetch(emailid, "(RFC822)")
            if resp != 'OK':
                print(f"Could not get a response from server for ID {emailid}")
                continue
        
            msg = email.message_from_bytes(data[0][1])
            
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode('utf-8', errors='ignore')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')

            email_info = {
                'Date': msg.get('Date'),
                'Subject': msg.get('Subject'),
                'From': msg.get('From'),
                'body': body
            }
            
            msg_id = msg.get('Message-ID', str(emailid))
            all_emails[msg_id] = email_info

        sign_in_link.logout()
        return all_emails

    except Exception as e:
        print(f"An error occurred during IMAP execution: {e}")
        return {}

# ==========================================
# NLP, STOPWORDS & VOCABULARY PREPROCESSING
# ==========================================

def load_stopwords(filepath='stopwords.txt'):
    """Loads stopwords from file, stripping newline characters."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip())
    return {'the', 'is', 'at', 'which', 'and', 'a', 'an', 'in', 'to', 'of', 'for', 'on', 'with'}

def parse_and_preprocess_emails(emails, stop_words):
    """
    Tokenizes email subject and body, applies stopword removal, and Porter stemming.
    """
    stemmer = PorterStemmer()
    documents = {}
    vocab = set()
    
    for msg_id, data in emails.items():
        subject = data.get('Subject', '') or ''
        body = data.get('body', '') or ''
        combined_text = f"{subject} {body}"
        
        words = re.findall(r'\b\w+\b', combined_text.lower())
        doc_terms = []
        
        for word in words:
            if word not in stop_words:
                stemmed_word = stemmer.stem(word)
                doc_terms.append(stemmed_word)
                vocab.add(stemmed_word)
                
        documents[msg_id] = doc_terms

    return documents, list(vocab)

# ==========================================
# VECTOR SPACE MODEL & TF-IDF MATRIX (NUMPY)
# ==========================================

def build_tfidf_matrix(documents, vocab):
    """
    Constructs a vectorized TF-IDF matrix for the email corpus using NumPy and L2 normalization.
    """
    num_docs = len(documents)
    vocab_size = len(vocab)
    
    vocab_to_idx = {term: idx for idx, term in enumerate(vocab)}
    msg_ids = list(documents.keys())
    
    tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float64)
    document_frequency = np.zeros(vocab_size, dtype=np.float64)
    
    print("Building TF-IDF matrix for email corpus...")
    for i, msg_id in enumerate(msg_ids):
        term_counts = Counter(documents[msg_id])
        
        for term, count in term_counts.items():
            if term in vocab_to_idx:
                idx = vocab_to_idx[term]
                tf_matrix[i, idx] = 1 + math.log(count)
                document_frequency[idx] += 1

    idf = np.log(num_docs / np.maximum(document_frequency, 1))
    tfidf_matrix = tf_matrix * idf
    
    manual_norms = np.zeros((num_docs, 1), dtype=np.float64)
    for i in range(num_docs):
        sum_squares = sum(tfidf_matrix[i, j] ** 2 for j in range(vocab_size))
        norm = math.sqrt(sum_squares)
        manual_norms[i, 0] = norm if norm > 0 else 1.0
    tfidf_matrix = tfidf_matrix / manual_norms
    
    return tfidf_matrix, vocab_to_idx, msg_ids

def process_query(query, vocab_to_idx, stop_words):
    """Converts a raw string query into a normalized TF vector."""
    stemmer = PorterStemmer()
    vocab_size = len(vocab_to_idx)
    query_vec = np.zeros(vocab_size, dtype=np.float64)
    
    words = re.findall(r'\b\w+\b', query.lower())
    processed_terms = []
    for word in words:
        if word not in stop_words:
            processed_terms.append(stemmer.stem(word))
            
    term_counts = Counter(processed_terms)
    for term, count in term_counts.items():
        if term in vocab_to_idx:
            idx = vocab_to_idx[term]
            query_vec[idx] = 1 + math.log(count)
            
    q_sum_sq = sum(val ** 2 for val in query_vec)
    q_norm = math.sqrt(q_sum_sq)
    if q_norm > 0:
        query_vec = query_vec / q_norm
        
    return query_vec

def search_emails(query_vec, tfidf_matrix, msg_ids, top_k=5):
    """Computes cosine similarity via dot product and returns ranked results."""
    similarities = tfidf_matrix.dot(query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0:
            results.append((msg_ids[idx], score))
            
    return results

# ==========================================
# GOOGLE SHEETS EXPORT
# ==========================================

def export_to_google_sheets(all_emails, credentials_file, sheet_name):
    """Exports email corpus data into a pandas DataFrame and syncs it with Google Sheets."""
    if not all_emails:
        print("No emails to export.")
        return

    df_data = []
    for msg_id, data in all_emails.items():
        df_data.append({
            'Message-ID': msg_id,
            'Subject of Email': data.get('Subject'),
            'Date of Email': data.get('Date'),
            'From': data.get('From'),
            'Body of Email': data.get('body')
        })
    
    df = pd.DataFrame(df_data)

    try:
        gc = pygsheets.authorize(service_file=credentials_file)
        sh = gc.open(sheet_name)
        wks = sh[0]
        wks.set_dataframe(df, (1, 1))
        print(f"Successfully exported {len(df)} emails to Google Sheet: {sheet_name}")
    except Exception as e:
        print(f"Google Sheets API integration failed ({e}). Exporting to local CSV instead.")
        df.to_csv("exported_emails.csv", index=False)
        print("Data successfully saved to 'exported_emails.csv'.")

# ==========================================
# INTERACTIVE CLI SEARCH LOOP
# ==========================================

def user_input_search(tfidf_matrix, vocab_to_idx, msg_ids, all_emails, stop_words):
    """Allows ranked interactive TF-IDF search queries across the loaded email corpus."""
    while True:
        user_input = input("\nEnter search query (or type 'quit' to exit): ").strip()
        if user_input.lower() == 'quit':
            print("Terminating program...")
            break
        if not user_input:
            continue
            
        query_vec = process_query(user_input, vocab_to_idx, stop_words)
        results = search_emails(query_vec, tfidf_matrix, msg_ids, top_k=5)
        
        if not results:
            print("No matching documents found.")
        else:
            print("\nTop Matching Emails:")
            for rank, (msg_id, score) in enumerate(results, 1):
                email_data = all_emails.get(msg_id, {})
                print(f"\n{rank}. Match Score: {score:.4f}")
                print(f"   Subject: {email_data.get('Subject')}")
                print(f"   Date: {email_data.get('Date')}")
                print(f"   Snippet: {(email_data.get('body') or '')[:200]}...")

# ==========================================
# EXECUTION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    username, password, recipient = user_prompt()
    
    print("Connecting to IMAP and fetching emails...")
    all_emails = fetch_emails(username, password, recipient)
    
    if all_emails:
        print("Loading stopwords and preprocessing emails...")
        stop_words = load_stopwords('stopwords.txt')
        documents, vocab = parse_and_preprocess_emails(all_emails, stop_words)
        
        print("Constructing Vector Space TF-IDF Matrix...")
        tfidf_matrix, vocab_to_idx, msg_ids = build_tfidf_matrix(documents, vocab)
        
        print("Exporting data to Google Sheets...")
        export_to_google_sheets(all_emails, GOOGLE_CREDENTIALS_FILE, GOOGLE_SHEET_NAME)
        
        user_input_search(tfidf_matrix, vocab_to_idx, msg_ids, all_emails, stop_words)
    else:
        print("No emails retrieved. Program exiting.")
