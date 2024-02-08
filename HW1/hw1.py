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

def lienar_regression(word, count):
    # word = np.array(word)
    count = np.array(count)
    word = np.arange(1, len(count) + 1)
    # Perform log transformation
    log_word = np.log(word)
    log_count = np.log(count)

    # Perform linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = linregress(log_word, log_count)

    # Plot original data and regression line on log-log scale
    plt.figure(figsize=(8, 6))
    plt.scatter(word, count, color='blue', label='Data')
    plt.plot(word, np.exp(intercept) * word**slope, color='red', label='Linear Regression')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.title('Linear Regression on Log-Log Scale')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Slope is ", slope)
    
def part_one():
    path = os.getcwd() + "/yelp"
    tokens = {}
    TFF_count = {}
    DF_count = {}
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    #Loop through each yelp file
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        #Loop through each review in the yelp _ file
        for review in reviews:
            doc = review['Content']
            doc_ID = review['ReviewID']
            tokens = word_tokenize(doc) #token the review content
            #Loop through each token and normal it
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif is_integer(token):
                    token = "NUM"
                elif is_stop_word(token):
                    continue
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                TFF_count[token] = 1 + TFF_count.get(token, 0)
                if token in DF_count and DF_count[token] is not None:
                    DF_count[token].add(doc_ID)
                else:
                    DF_count[token] = set()
                    DF_count[token].add(doc_ID)

    # with open("output.txt", "w") as file:
    #     sys.stdout = file
    #     print(TFF_count)
    #     print("==================================================================")
    #     print(DF_count)

    # graph TFF
    sorted_TFF_count = dict(
        sorted(TFF_count.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_TFF_count.keys())
    count = list(sorted_TFF_count.values())
    # plt.figure(figsize=(8, 6))
    # plt.loglog(word, count, marker='o', linestyle='-', color='b')
    # # Set the x-axis to log scale
    # plt.xscale('log')
    # plt.xlabel('X-axis (word)')
    # plt.ylabel('Y-axis (count)')
    # plt.title('TTF graph')
    # plt.show()
    
    lienar_regression(word,count)
    # with open("TFF_count.txt", "w") as file:
    #     for c in count:
    #         file.write(str(c) + "\n")
    
    # graph DF
    for key in DF_count:
        DF_count[key] = len(DF_count[key])
    sorted_DF_count = dict(
        sorted(DF_count.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_DF_count.keys())
    count = list(sorted_DF_count.values())

    lienar_regression(word,count)
    
    # plt.figure(figsize=(8, 6))
    # plt.loglog(word, count, marker='o', linestyle='-', color='b')

    # # Set the x-axis to log scale
    # plt.xscale('log')

    # plt.xlabel('X-axis (word)')
    # plt.ylabel('Y-axis (count)')
    # plt.title('DF graph')
    # plt.show()



def part_two_es():
    # query
    queries = ["general chicken", "fry chicken", "bbq sandwich", "mash potato", "grill shrimp salad",
               "lamb shank", "pepperoni pizza", "friend staff", "grill cheese"]
    #  ES search
    # indexing documents
    es = Elasticsearch('https://localhost:9200', basic_auth=("elastic",
                       "B_mJvwy5xsSJdNwjcAx9"), verify_certs=False)
    path = os.getcwd() + "/yelp"
    yelp_store_list = os.listdir(path)
    index_name = "text_index"
    documents = []
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        # print("Current review: ", reviews)
        for review in reviews:
            doc = review['Content']
            doc_ID = review['ReviewID']
            doc_e = {
                '_index': index_name,
                '_id': doc_ID,
                'review': doc
            }
            documents.append(doc_e)
    helpers.bulk(es, documents)
    es_TTF = {}
    es_DF = {}
    with open("output.txt", 'w') as file:
        pass
    
    start_time = time.time()

    for query in queries:
        # Construct the query using the correct syntax
        resp = es.search(index=index_name, body={
                         "query": {"match": {"review": query}}})
        # Write the response to a file
        with open("output.txt", "a") as file:
            sys.stdout = file
            print("Searching for ", query)
            print("Got %d Hits:\n" % resp['hits']['total']['value'])

    end_time = time.time()
    with open("output.txt", 'a') as file:
        sys.stdout = file
        total_time = end_time - start_time
        print(f"==========Total running time is {total_time} ==========")

    # Find TTF
    for query in queries:
        # Construct the query using the correct syntax
        resp = es.search(index=index_name, body={
                         "query": {"match": {"review": query}}})
        # Iterate over hits and get term vectors for each document
        for hit in resp['hits']['hits']:
            doc_id = hit['_id']
            term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["review"])
            # Extract TF and added it to the dictionary
            terms_info = term_vectors["term_vectors"]["review"]["terms"]
            for term, stats in terms_info.items():
                tf = stats["term_freq"]
                es_TTF[term] = tf + es_TTF.get(term,0)
        
    with open("output.txt", 'a') as file:
        sys.stdout = file
        print("==========TFF==========")
        print(es_TTF)
        
        
    #Go over every term in the return query of document and search again and record those document
    for term, stats in es_TTF.items():
        df = es.search(index=index_name, body={"query": {"match": {"review": term}}})['hits']['total']['value']
        es_DF[term] = df + es_TTF.get(term,0)
    
    with open("output.txt", 'a') as file:
        sys.stdout = file
        print("==========DF==========")
        print(es_DF)
                
    #graph TFF
    sorted_TF_count = dict(
        sorted(es_TTF.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_TF_count.keys())
    count = list(sorted_TF_count.values())

    plt.figure(figsize=(8, 6))
    plt.loglog(word, count, marker='o', linestyle='-', color='b')

    # Set the x-axis to log scale
    plt.xscale('log')

    plt.xlabel('X-axis (word)')
    plt.ylabel('Y-axis (count)')
    plt.title('ES_TFF graph')
    plt.show()
    
    #graph DF
    sorted_DF_count = dict(
        sorted(es_DF.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_DF_count.keys())
    count = list(sorted_DF_count.values())

    plt.figure(figsize=(8, 6))
    plt.loglog(word, count, marker='o', linestyle='-', color='b')

    # Set the x-axis to log scale
    plt.xscale('log')

    plt.xlabel('X-axis (word)')
    plt.ylabel('Y-axis (count)')
    plt.title('ES_DF graph')
    plt.show()
    
                
    

def part_two_invered_index():
    
    path = os.getcwd() + "/yelp"
    inverted_index = {}
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    #loop through all the yelp data
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        #loop through each review item in each data
        for review in reviews:
            doc = review['Content']
            doc_ID = review['ReviewID']
            tokens = word_tokenize(doc)
            #Normalize each word 
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif is_integer(token):
                    token = "NUM"
                elif is_stop_word(token):
                    continue
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                #For each word, record the document that contain it
                if token in inverted_index and inverted_index[token] is not None:
                    inverted_index[token].add(doc_ID)
                else:
                    inverted_index[token] = set()
                    inverted_index[token].add(doc_ID)
        
    #make query
    queries = ["general chicken", "fry chicken", "bbq sandwich", "mash potato", "grill shrimp salad",
               "lamb shank", "pepperoni pizza", "friend staff", "grill cheese"]
    
    with open ("output_inverted_index.txt", "w") as file:
        pass
    start_time = time.time()
    result = {}
    for query in queries:
        split_query = query.split(" ")
        word_one = stemmer.stem(split_query[0])
        word_two = stemmer.stem(split_query[1])
        print(word_one)
        print(word_two)
        try:
            first_result = inverted_index[word_one]
            second_result = inverted_index[word_two]
            result = first_result.intersection(second_result)
            print(f"Query for {query} and found {len(result)} documents")
        except KeyError as e:
            print(f"Query for {query} raised have 0 hits")
        
    end_time = time.time()
    with open ("output_inverted_index.txt", "a") as file:
        sys.stdout = file
        print(f"Total time is {end_time - start_time}")

def main():
    # part_one()
    # part_two_es()
    part_two_invered_index()


if __name__ == "__main__":
    main()
