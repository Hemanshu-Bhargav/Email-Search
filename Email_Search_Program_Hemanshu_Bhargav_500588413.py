# First, extract a user's gmail for email messages which will form the corpus
# This corpus will then be exported to Google Sheets
# As explained in the proposal, the intended usage of this program 
# is to "bridge the gap" that exists for Google Add-on development
# Although, search within Google exists, if any GMail account users wishes
# to install an add-on for increased functionality, the onus is on the developer
# of that add-on to implement a reliable search program which is compatible
# with Google's services. This program aims to be a portable solution.
# Note: Understandably, python's execution is not as fast as say Java's, due 
# to the differences of compiled and interpreted languages, but because assignment
# one and two were written in Python, this project is as well (due to time constraints)

# As add-ons can only use Google sheets for their database and because parsing attachments
# is both out of the scope of this project and ethically questionable, attachments are ignored 

# The imaplib library connects python scripts to any recognized 
# email server. Due to the "business case" outlined in the proposal, only Google accounts 
# are used

import email
import imaplib
import json
import os
import re
import getpass
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
import pygsheets

# ==========================================
# CONFIGURATION
# ==========================================
# Replace this with your service account file path or use environment variables
GOOGLE_CREDENTIALS_FILE = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "path/to/your/service_account.json"
)
GOOGLE_SHEET_NAME = "CPS 842 Project V1"

# ==========================================
# IMAP AUTHENTICATION & EMAIL FETCHING
# ==========================================

# IR app prompts user for their credentials
def user_prompt():
    username = input("Please enter your Gmail username: ")
    password = getpass.getpass("Enter your password: ")
    recipient = input("Please enter whose emails you'd like to store (e.g., sender@gmail.com): ")
    return username, password, recipient

def fetch_emails(username, password, recipient):
    """Connects to Gmail via IMAP and extracts emails from a specific sender."""
    try:
        # Pass Google's server link as a parameter
        sign_in_link = imaplib.IMAP4_SSL("imap.gmail.com")
        sign_in_link.login(username, password)
        sign_in_link.select("INBOX")
        
        resp, items = sign_in_link.search(None, 'FROM', recipient)
        if resp != 'OK' or not items[0]:
            print("No messages found or failed to search inbox.")
            return {}

        all_emails = {}
        for emailid in items[0].split():
            resp, data = sign_in_link.fetch(emailid, "(RFC822)") # change to read-only version later
            if resp != 'OK':
                print(f"Could not get a response from server for ID {emailid}")
                continue
        
            msg = email.message_from_bytes(data[0][1])
            
            # Extract body safely
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
# NLP & INVERTED INDEX PIPELINE
# ==========================================

def load_stopwords():
    """Loads stopwords from file if available, otherwise uses a standard set."""
    if os.path.exists('stopwords.txt'):
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f)
    return {'the', 'is', 'at', 'which', 'and', 'a', 'an', 'in', 'to', 'of', 'for', 'on', 'with'}

def clean_and_tokenize(text):
    """Tokenizes text into lowercase words, stripping punctuation."""
    if not text:
        return []
    # take all arguments and tokenize using .split() or regex equivalent
    return re.findall(r'\b\w+\b', text.lower())

# Create dictionaries of body and term frequencies
# @param body, subject refers to body and subject text
# the only columns which are relevant and are used for the inverted index
def build_inverted_index(emails):
    """Builds a stemmed inverted index mapping keywords to email Message-IDs."""
    stemmer = PorterStemmer()
    stopwords_set = load_stopwords() # below 2 line snippet retrieved from https://pythonprogramming.net/stop-words-nltk-tutorial/ on October 4, 2019
    inverted_index = {}
    
    for msg_id, email_data in emails.items():
        subject = email_data.get('Subject', '') or ''
        body = email_data.get('body', '') or ''
        combined_text = f"{subject} {body}"
        
        tokens = clean_and_tokenize(combined_text)
        
        for token in tokens:
            if token not in stopwords_set:
                # FIX THIS?? handled type conversion and single string issue cleanly here
                stemmed_word = stemmer.stem(token)
                if stemmed_word not in inverted_index:
                    inverted_index[stemmed_word] = set()
                inverted_index[stemmed_word].add(msg_id)
                
    return {word: list(msg_ids) for word, msg_ids in inverted_index.items()}

# ==========================================
# GOOGLE SHEETS & DATAFRAME INTEGRATION
# ==========================================

# Dataframes are reliable structures to store data which must be transformed into CSV files 
# (Since Google Sheets are spreadsheets, I treat them as any comma seperated sheet)
def export_to_google_sheets(all_emails, credentials_file, sheet_name):
    """Exports email corpus data into a pandas DataFrame and syncs it with Google Sheets."""
    if not all_emails:
        print("No emails to export.")
        return

    columns = ['Subject of Email', 'Date of Email', 'Body of Email', 'From', 'Message-ID']
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
        # Create google cloud API, service account, Google sheet and enable domain delegation prior to below authorization
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
# CLI INTERACTIVE SEARCH
# ==========================================

def user_input_search(all_emails):
    """Allows interactive search queries across the loaded email corpus."""
    stemmer = PorterStemmer()
    while True:
        searchword = input("\nPlease enter a search term (or type 'quit' to exit): ").strip()
        if searchword.lower() == "quit":
            print("You've terminated the program!")
            break
        
        stemmed_query = stemmer.stem(searchword.lower())
        found = False
        
        for msg_id, data in all_emails.items():
            body_text = (data.get('body') or '').lower()
            subject_text = (data.get('Subject') or '').lower()
            
            if stemmed_query in body_text or stemmed_query in subject_text or searchword.lower() in body_text:
                print(f"\n--- Match Found ---")
                print(f"Subject: {data.get('Subject')}")
                print(f"Date: {data.get('Date')}")
                print(f"Body Snippet: {data.get('body')[:300]}...")
                found = True
                
        if not found:
            print("Search term does not exist")

# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    username, password, recipient = user_prompt()
    
    print("Connecting to IMAP and fetching emails...")
    all_emails = fetch_emails(username, password, recipient)
    
    if all_emails:
        print("Constructing inverted index...")
        inverted_idx = build_inverted_index(all_emails)
        
        print("Exporting data to Google Sheets...")
        export_to_google_sheets(all_emails, GOOGLE_CREDENTIALS_FILE, GOOGLE_SHEET_NAME)
        
        user_input_search(all_emails)
    else:
        print("No emails retrieved. Program exiting.")
