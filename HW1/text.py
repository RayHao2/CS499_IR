import os
import json
from nltk.tokenize import word_tokenize
import nltk
from string import punctuation
from nltk.stem import PorterStemmer
import re
import sys
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
import time
from elasticsearch import helpers
import numpy as np
from scipy.stats import linregress

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_fraction(s):
    fraction_pattern = re.compile(r'^\d+/\d+$')
    return bool(fraction_pattern.match(s))


def remove_non_alphanumeric(input_string):
    # Use regular expression to keep only English letters and numbers
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result



def get_term_vectors(es, index_name, doc_id, field_name):
    term_vectors_query = {
        "fields": [field_name],
        "offsets": False,
        "positions": False,
        "term_statistics": True,
        "field_statistics": True
    }
    response = es.termvectors(
        index=index_name, doc_type="_doc", id=doc_id, body=term_vectors_query)
    term_statistics = response["term_vectors"][field_name]["terms"]
    return term_statistics


def is_stop_word(token):
    stop_words = set([
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
        'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the',
        'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with'
    ])

    # Convert the word to lowercase for case-insensitive comparison
    lowercase_word = token.lower()

    return lowercase_word in stop_words

reviews = ["I want geneRal chiCken", "GenEral chicken is the best"]
stemmer = PorterStemmer()
for review in reviews:
    tokens = word_tokenize(review)
    #Normalize each word 
    for token in tokens:
        print("token before ", token)
        token = remove_non_alphanumeric(token)
        print("token after remove_non_alpha ", token)
        
        if len(token) == 0:
            continue
        elif is_integer(token):
            token = "NUM"
        elif is_stop_word(token):
            continue
        else:
            token = stemmer.stem(token)
            token = token.lower()
        print("token After ", token)

