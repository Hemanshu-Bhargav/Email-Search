README

Files in this directory with sizes:
    	October 31 23:59 Assignment2CPS842_Bhargav_Ashraf   	Search Retrieval Program
	2668 October 31 stopwords.text        			alternative stop words list
	2187734 Jun 19 20:55 cacm.all             		 text of documents
        626 Jun 19 20:58 cite.info            			 key to citation info
                                             			   (the X sections in cacm.all)
       9948 Jun 19 20:55 qrels.text           			  relation giving
                                              			      qid did 0 0
                                              			  to indicate dument did is
                                              			  relevant to query qid
      13689 Jun 19 20:55 query.text           			  Original text of the query



Program Description

 It first creates an inverted index of searchable terms. The program prompts for a term. User inputs the term and program searches through the dictionary. 
It returns all the relevant documents where the search term occurs. It also returns a ranking of the search results by relevance weight. The is stemming done by 
mystemmer, which can be used to switch stemming on or off. The weighting scheme for td-idf used to determine this is 1+log of term. This weight is then compared to 
actual precision from document qrels.txt and returns the Mean Average Precision and R-Precision. The top-k retriveal method implemented is IDF threshold. Posting lists 
are ordered by document ID. The user is prompted infintely (or until memory crashes) for test queries, unless they type 'quit' to terminate the program.  

Program Execution

Run the program on the command line. It will prompt the user to input search term. When a term is entered, it will return all the relevant documents, 
ranked by their relevancy weight. It will also return the _____and the r-precision value of  the relevancy weight compared to the actual user judgement from 
the document qrels.text. The program will keep prompting for another search query and will keep doing so until 'quit; is entered, which will terminate the program.

Data Structures Used

Below lists the organization of the different parts of the corpus in their respective data structures
 Dictionaries are nested w/ other data structures for fast access
 dicDoc.keys = document id
 dicDoc.values = tuple of terms and term frequency in the document
 wordsDoc.keys = term
 wordsDoc.values = document frequency (documents the term occurs in)
 search_Terms = searchable index stored as list
 docsTFIDFvectors = dictionary consisting of document vectors whose values are numpy arrays equivalent to the lenght of corpus
 queryvec = numpy array to store tokenized query terms (equivalent to the lenght of corpus)
When designing any text retrieval system, performance is a key constraint, even more than in other user applications. The reason being, that the end-users of the application have vast information needs which change rapidly based on the results of their previous searches. Furthermore, any search system must be customizable, so that it can be updated and corrected after running searches. This requires a modular design principle, so that key components can be modified and updated when more efficient stemming/lemmatization algorithms become available. With the concepts of performance and modularity in mind, we chose the following data structures/algorithms:

Python Dictionary: In order to meet the requirements of dynamic key insertion and deletion, a hash data structure is conventionally used. Although Python has no data structures named using 'hash', Python dictionaries are technically implemented as hashmaps behind the scenes. Python dictionaries were also chosen for another purpose, performance. A Python 'dict' has an average look-up time of O(n), which would satisfy most users' expectations, had the system been released for public use. In the best case scenario, 'dicts' have m*n time complexity . Finally, in their worst case scenario, 'dicts' run m * n * n.  
Furthermore, dictionaries in Python can be nested. This allows for dynamic access of not just the document frequency and terms, but also the information found in the posting list. By creating two dictionaries to hold all of the information required by both the posting list and dictionary, extraction of information becomes simply a means of variable transfer. No information loss occurs (so a one-to-one correspondence is ensured)

Python List: For our posting list, and even for our intermediary document storage, we opted for Python lists, due to their mutability and performance. Python lists also are great for space conscious programmers such as ourselves, as lists have an average of O(n) space complexity.

Python Natural Language Text Processing Toolkit & Porter Stemmer Algorithm: Python has arguably the strongest natural language library support of any programming language. This library includes standard modules which have been tried and tested for lemmatization, stop word trimming, tokenizing and stemming. For the purposes of this assignment, we only used the Porter Stemmer algorithm. The Porter Stemmer algorithm from Python’s NLTK library was used over the standard published version for several reasons. This version runs the “.NLTK” extension by default, which is a modified version of the original Porter Stemmer algorithm, including updates but still maintaining all core functionalities. The original algorithm, which was published by the author in the 1980s, has been deprecated since, but is also available as an option with the NLTK version, along with the updated extension and their default extension which we opted for. 
Numpy Array: Allows operations on random access. For the purposes of matching indexes between the TF/IDF intermediate vectors before and after normalization, numpy arrays have their own unique methods which allowed a one-to-one mapping.