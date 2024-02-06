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
# Get all the content from json file to a txt file
    # path = os.getcwd() + "/yelp"
    # yelp_store_list = os.listdir(path)
    # for yelp_store in yelp_store_list:
    #     print("Current file: ", yelp_store)
    #     store_id = yelp_store.strip(".json")
    #     store_path = path + "/" +yelp_store
    #     with open (store_path, 'r') as file:
    #         data = json.load(file)
    #     reviews = data["Reviews"]
    #     for review in reviews:
    #         new_file = os.getcwd() + "/yelp_content/" + store_id + ".txt"
    #         with open (new_file, "a") as file:
    #             file.write(review["Content"])

    
    
    #query 
    queries = ["general chicken", "fry chicken", "bbq sandwich", "mash potato", "grill shrimp salad",
            "lamb shank", "pepperoni pizza", "friend staff", "grill cheese"
            ]
    #  ES search    
    # indexing documents
    es = Elasticsearch('https://localhost:9200', basic_auth=("elastic", "B_mJvwy5xsSJdNwjcAx9"), verify_certs=False)
    path = os.getcwd() + "/yelp_content"
    yelp_store_list = os.listdir(path)
    for index in range(len(yelp_store_list)):
        store_id = yelp_store_list[index].strip(".txt")
        file_path = os.path.join(path, yelp_store_list[index])
        with open(file_path, 'r') as file:
            document_content = file.read()
        index_name = f"yelp_{store_id.lower()}"  
        doc = {"content": document_content}
        resp = es.index(index=index_name, id=index, document=doc)
        # print(resp)
        
    print("Done indexing and starting searching")
    start_time = time.time()
    #Search each query by making a searfch query to ES
    for query in queries:
        print(f"Searching for: {query}")
        index_pattern = "yelp_*"
        search_query = {
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        #result obtain
        result = es.search(index=index_pattern, body=search_query)
        #Decode the result
        for hit in result['hits']['hits']:
            #write the document retrived in a file
            with open("ESSearch_output.txt", "a") as file:
                output = f"Searching for {query} Score: {hit['_score']}, Index: {hit['_index']}, Document ID: {hit['_id']}\n"
                file.write(output)
        #line break
        with open("ESSearch_output.txt", "a") as file:
            file.write("=================================\n")
            
            
    
    #record end time
    # end_time = time.time()
    # with open("ESSearch_output.txt", "a") as file:
    #     file.write(f"Total time used in ES is {end_time - start_time}")
    
    
    #Log Log Curve for ES search
    # TFF = {}
    # DF = {}
    # with open("ESTFFandDF.txt", "r") as file:
    #     for line in file:
    #         tokens = line.split(", ")
    #         # Extract values from tokens
    #         term = tokens[1].split(": ")[1]
    #         tf = int(tokens[2].split(": ")[1])
    #         df = int(tokens[3].split(": ")[1])
    #         # print(f"Term:{term} TF: {tf}, DF{df}")
    #         TFF[term] = tf
    #         DF[term] = df

    # with open("DF.txt", 'w') as file:
    #     sys.stdout = file
    #     print(DF)
    
    # # Sort TFF
    # sorted_DF = dict(
    #     sorted(DF.items(), key=lambda item: item[1], reverse=True))
    # word = list(sorted_DF.keys())
    # count = list(sorted_DF.values())

    # plt.figure(figsize=(8, 6))
    # plt.loglog(word, count, marker='o', linestyle='-', color='b')

    # # Set the x-axis to log scale
    # plt.xscale('log')

    # plt.xlabel('X-axis (word)')
    # plt.ylabel('Y-axis (count)')
    # plt.title('ES search DF graph')
    # plt.show()
    
    
    # Building Bag of word
    #Tokenize and count the DF for each word
    Bag_of_word = {}
    path = os.getcwd() + "/yelp_content"
    yelp_store_list = os.listdir(path)
    stemmer = PorterStemmer()
    for yelp_store in yelp_store_list:
        store_id = yelp_store.strip(".txt")
        path = os.getcwd() + "/yelp_content/" + yelp_store
        with open(path, 'r') as file:
            contents = file.read()
        tokens = word_tokenize(contents)
        for token in tokens:
            # strip all the non-English letters and digits
            token = remove_non_alphanumeric(token)
            if len(token) == 0:
                continue
            elif is_integer(token):
                token = "NUM"
            else:
                token = stemmer.stem(token)
                token = token.lower()

        

        
def main():
    # part_one()
    part_two()


if __name__ == "__main__":
    main()
