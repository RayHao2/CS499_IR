from nltk.util import bigrams
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import json
import time
import re
def remove_non_alphanumeric(input_string):
    # Use regular expression to keep only English letters and numbers
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    
token = "yelpcombizphotoscbw85"
token = remove_non_alphanumeric(token)
# stemmer = PorterStemmer()
# if len(token) == 0:
#     print(0)
# elif is_integer(token):
#     token = "NUM"
# else:
#     token = stemmer.stem(token)
#     token = token.lower()
# print(token)