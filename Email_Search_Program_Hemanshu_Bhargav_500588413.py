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
import email, imaplib, json, os, re, getpass, math, sys, time
# import os, re, time, sys
import numpy as np
from nltk.stem import PorterStemmer
import pandas as pd
# As add-ons can only use Google sheets for their database and because parsing attachments
# is both out of the scope of this project and ethically questionable, attachments are ignored 

# The imaplib library connects python scripts to any recognized 
# email server. Due to the "business case" outlined in the proposal, only Google accounts 
# are used

# IR app prompts user for their credentials
def userprompt():
    username = input("Please enter your gmail username: ")
    password = getpass.getpass("Enter your password:  ")
    receipient = input("Please which receipient's emails you'd like to store: ")
    return username, password, receipient
username, password, receipient = userprompt()
def mainsignin(username, password, receipient):
    # Pass Google's server link as a parameter
    sign_in_link = imaplib.IMAP4_SSL("imap.gmail.com")
    sign_in_link.login(username, password)
    sign_in_link.select("INBOX")
    resp, items = sign_in_link.search(None, 'FROM', receipient)
    
    all_emails = {}
    for emailid in items[0].split():
        resp, data = sign_in_link.fetch(emailid, "(RFC822)") #change to read-only version later
        if resp != 'OK':
            print("Could not get a response from server")
            break
    
        msg = email.message_from_bytes(data[0][1])
        # email_info = {**msg} # Adds all data to the resulting dictionary
        email_info = {
            'Date': msg['Date'],
            'Subject': msg['Subject'],
            'From': msg['From'],
        }
        if msg.is_multipart():
            for payload in msg.get_payload():
                email_info['body'] = payload.get_payload()
        else:
            email_info['body'] = msg.get_payload()
        all_emails[msg['Message-ID']] = email_info
   # print(all_emails)
    return all_emails
    sign_in_link.logout()
def userinput(all_emails):
    while(1):
        searchword = input("Please enter a search term: ")
        if searchword == "quit":
            print("You've terminated the program!")
            break
        else:
            #take all arguments and tokenize using .split()
            searchword = searchword.split()
            if searchword not in all_emails.values():
                print("Search term does not exist")
            else:
                for item in searchword: #for subject in subject.values():
                    for d in all_emails.values():
                        if item in d['body']:
                            print(d['body'])
                            print(d['Subject'])
all_emails = mainsignin(username, password, receipient)
userinput(all_emails)

#print(json.dumps(all_emails, indent=4))
#print(get_bodies(all_emails))

        
def get_column(emails, column):
    """ Returns a list of email bodies

    param emails: Dictionary of emails
    """
    value =  [d[column] for d in emails.values()]
   # print(type(value))
    return value
# Extract columns to begin inverted index construction and to export to Google Sheets
date_column = get_column(all_emails, 'Date')
body_column= get_column(all_emails, 'body')
subject_column= get_column(all_emails, 'Subject')

# Create dictionaries of body and term frequencies
# @param body, subject refers to body and subject text
# the only columns which are relevant and are used for the inverted index
def invertedindex(body, subject):
    for word in body, subject:
        word = (str(word)).split()
        corpus = word.split()
        #print(corpus)
    return corpus
#corpus = invertedindex(body_column, subject_column)
def stopwords(corpus):
    stop = open('stopwords.txt', 'r')
    criteria = set(stop)
    # below 2 line snippet retrieved from https://pythonprogramming.net/stop-words-nltk-tutorial/ on October 4, 2019
    filtered_words = [w for w in corpus if w not in criteria]
    return filtered_words

def mystemmer(stopped_words):
    stemming = PorterStemmer()
    token = str(stopped_words) #FIX THIS?? causes search_terms to be one string of corpus
    #for this reason search_Terms was used
    stemmed_terms = stemming.stem(token)
    return stemmed_terms

#stopped_words = stopwords(all_emails) #change to corpus
#search_Terms = mystemmer(stopped_words)

# Dataframes are reliable structures to store data which must be transformed into CSV files 
# (Since Google Sheets are spreadsheets, I treat them as any comma seperated sheet)
def export_to_Google_Sheets(subject_column, body_column, date_column):
    columns = ['Subject of Email','Date of Email', 'Body of Email']
    df = pd.DataFrame(index = [0] , columns = columns)
    df.loc[:, 'Subject of Email'] = subject_column
    df.loc[:, 'Date of Email'] = date_column
    df.loc[:, 'Body of Email'] = body_column

## Grab all email messages
#
#sign_in_link.list()
#
## Choose the inbox to retrieve emails from. Of course, we can pass
## any sub-folder we wish, by simply prompting the user and passing the
## string we save. (DEMO THIS LATER)
#sign_in_link.select("INBOX")
#receipient = input("Please which receipient's emails you'd like to store: ")
#resp, items = sign_in_link.search(None, 'FROM', receipient)
#items = items[0].split()  
#body = ""
#for emailid in items:
#    resp, data = sign_in_link.fetch(emailid, "(RFC822)") #change to read-only version later
#    data.append(data[0][1]) #appends all emails instead of retrieving only latest
#    if (resp != 'OK'):
#        print("Could not get a response from server")
#        break
# # Gets body with all network details when printed "as_string"  msg1 = email.message_from_bytes(data[0][1])
## Get Subject and Date 
#    for response_part in data:
#        if isinstance(response_part, tuple):
#          msg = email.message_from_bytes(response_part[1]) #equivalent to from_string, but didn't work, possibly due to Python 3
#          varSubject = msg['subject']  
#          varDate = msg['date']
#         #STORE SCRAPED EMAIL DATA IN DATAFRAME, rather than dictionary to export as Google Sheet
#          columns = ['Subject of Email','Date of Email', 'Body of Email']
#          df = pd.DataFrame(index = [0] , columns = columns)
#          df.loc[:, 'Subject of Email'] = varSubject
#          df.loc[:, 'Date of Email'] = varDate
#        #  print("Subject:", varSubject, "This email was sent on", varDate)
#          # Extract email body without the network details (body extraction is not a one-liner)
#          messageMainType = msg.get_content_maintype()
#          if messageMainType == 'multipart':
#              for part in msg.get_payload(): #think of payload as the body
#                  if part.get_content_maintype() == 'text':
#                      body = part.get_payload()
#                  body = ""
#        elif messageMainType == 'text':
#            body = msg.get_payload()
#        df.loc[:, 'Body of Email'] = body
#        print(df)
#def userinput(all_emails,)
#        while(1):
#            searchword = input("Please enter a search term: ")
#            if searchword == "quit":
#                 print("You've terminated the program!")
#                 break
#            else:
#                #take all arguments and tokenize using .split()
#                searchword = searchword.split()
#                 #for each token in "argv", repeat normalization procedure
#                for item in searchword:
#                    for subject in subject.values():
#                        for value in body.values():
#                            if item = body[value]:
#                                print()
#        
#       # print(df['Body of Email'])
#        # BELOW LOOP DOES NOT WORK, FIX!
#        # initialize dictionary to store emails and frequencies
## Use nested or singular dictionaries (nested dictionary approach is both efficient and modular, 
## so making it an optimal storage strategy)
#        emaildict = {} 
#        for key in emaildict.keys():
#            emaildict[key] = varSubject
#            print(emaildict.keys())
#        for value in emaildict.values():
#            emaildict[value] = body
#        print(emaildict.values())
#import pygsheets
## Create google cloud API, service account, Google sheet and enable domain delegation prior to below
##authorization
#gc = pygsheets.authorize(service_file='C:/Users/Hemanshu/Desktop/inbound-lattice-260111-1786ae9e0f23.json')
##open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
##sh = gc.open('CPS 842 Project V1')
##
###select the first sheet 
##wks = sh[0]
##
###update the first sheet with df, starting at cell B2. 
##wks.set_dataframe(df,(1,1))
##wks.set_dataframe(df,(0,0))
##wks.set_dataframe(df,(0,1))
##wks.set_dataframe(df,(1,0))
##wks.set_dataframe(df,(1,2))
##wks.set_dataframe(df,(2,0))
##wks.set_dataframe(df,(2,1))
##wks.set_dataframe(df,(2,2))
#sign_in_link.logout()
