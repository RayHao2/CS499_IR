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

def get_term_vectors(es,index_name, doc_id, field_name):
    term_vectors_query = {
        "fields": [field_name],
        "offsets": False,
        "positions": False,
        "term_statistics": True,
        "field_statistics": True
    }
    response = es.termvectors(index=index_name, doc_type="_doc", id=doc_id, body=term_vectors_query)
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
def part_one():
    path = os.getcwd() + "/yelp"
    tokens = {}
    TFF_count = {} 
    DF_count = {}
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        # print("Current review: ", reviews)
        for review in reviews:
            doc = review['Content']
            doc_ID = review['ReviewID']
            tokens = word_tokenize(doc)
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
                TFF_count[token] = 1 + TFF_count.get(token,0)
                if token in DF_count and DF_count[token] is not None:
                    DF_count[token].add(doc_ID)
                else:
                    DF_count[token] = set()
                    DF_count[token].add(doc_ID)
                    
    with open("output.txt", "w") as file:
        sys.stdout = file
        print(TFF_count)
        print("==================================================================")
        print(DF_count)
                    
    #graph TFF
    sorted_TFF_count = dict(
    sorted(TFF_count.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_TFF_count.keys())
    count = list(sorted_TFF_count.values())

    plt.figure(figsize=(8, 6))
    plt.loglog(word, count, marker='o', linestyle='-', color='b')

    # Set the x-axis to log scale
    plt.xscale('log')

    plt.xlabel('X-axis (word)')
    plt.ylabel('Y-axis (count)')
    plt.title('TTF graph')
    plt.show()

    #graph DF
    for key in DF_count:
        DF_count[key] = len(DF_count[key])
    sorted_DF_count = dict(
    sorted(DF_count.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_DF_count.keys())
    count = list(sorted_DF_count.values())

    plt.figure(figsize=(8, 6))
    plt.loglog(word, count, marker='o', linestyle='-', color='b')

    # Set the x-axis to log scale
    plt.xscale('log')

    plt.xlabel('X-axis (word)')
    plt.ylabel('Y-axis (count)')
    plt.title('DF graph')
    plt.show()
    


def part_two():
    #query 
    queries = ["general chicken", "fry chicken", "bbq sandwich", "mash potato", "grill shrimp salad",
            "lamb shank", "pepperoni pizza", "friend staff", "grill cheese"]
    #  ES search    
    # indexing documents
    es = Elasticsearch('https://localhost:9200', basic_auth=("elastic", "xq40waSiYVhL1nf22tgO"), verify_certs=False)
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
                '_index' : index_name,
                '_id' : doc_ID,
                'review' : doc
            }
            documents.append(doc_e)
    helpers.bulk(es,documents)

    for query in queries:
        # Construct the query using the correct syntax
        resp = es.search(index = index_name, body={"query": {"match": {"review": query}}})
        # Write the response to a file
        with open("output.txt", "a") as file:
            sys.stdout = file 
            print(resp)
            print("================================\n")
            # file.write("Got %d Hits:\n" % resp['hits']['total']['value'])
            # for hit in resp['hits']['hits']:
            #     file.write("%(timestamp)s %(author)s: %(text)s\n" % hit["_source"])
            # file.write("================================\n")


    #Do inverted index for BoW
            
    


        

        
def main():
    # part_one()
    part_two()


if __name__ == "__main__":
    main()
