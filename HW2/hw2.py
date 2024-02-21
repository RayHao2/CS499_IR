import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import sys
import pickle
import heapq
from nltk.util import bigrams
import time
import random
import numpy as np
import math

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def remove_non_alphanumeric(input_string):
    # Use regular expression to keep only English letters and numbers
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result


def process_documents():
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    tokens_count = {}
    unigram_percentage = {}
    stemmer = PorterStemmer()
    total_tokens = 0
    yelp_store_list = os.listdir(path)
    bigram_freq_dict = {}
    #unigram
    
    #Prepare documents
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        #Loop through each review in the yelp _ file
        for review in reviews:
            doc = review['Content']
            tokens = word_tokenize(doc) #token the review content
            normaled_token = []
            #Loop through each token and normal it
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif is_integer(token):
                    token = "NUM"
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                tokens_count[token] = 1 + tokens_count.get(token, 0)
                normaled_token.append(token)
            bigram_list = list(bigrams(normaled_token))
            for bigram in bigram_list:
                bigram_freq_dict[bigram] = 1 + bigram_freq_dict.get(bigram,0)


    #find total amount of token in the dic
    for token in tokens_count:
        total_tokens += tokens_count[token]
    #calulate unigram percentage
    for token in tokens_count:
        count = tokens_count[token]
        unigram_percentage[token] = count / total_tokens

    
    #store to local file
    with open("tokens_count.pkl", "wb") as f:
        pickle.dump(tokens_count,f)
    with open("unigram_percentage.pkl", "wb") as f:
        pickle.dump(unigram_percentage,f)
    with open("total_tokens.txt", "w") as f:
        f.write(str(total_tokens))
    with open('bigram_freq_dict.pkl', 'wb') as f:
        pickle.dump(bigram_freq_dict, f)


#find number of w_i-1 given the w_i:
def count_gien_word_given_query_word(bigram_frquency_dic):
    count_gien_word_given_query_word = {}
    for bigram in bigram_frquency_dic:
        count_gien_word_given_query_word[bigram[1]] = 1 + count_gien_word_given_query_word.get(bigram[1], 0)
        # print("found one for ", token)
    
    # print(count_gien_word_given_query_word)
    with open('count_gien_word_given_query_word.pkl', 'wb') as f:
        pickle.dump(count_gien_word_given_query_word, f)

#find number of bigram starting with w_i-1
def count_big_freq_of_given_word(bigram_frquency_dic):
    count_big_freq_of_given_word = {}
    for bigram in bigram_frquency_dic:
        count_big_freq_of_given_word[bigram[0]] = bigram_frquency_dic[bigram] + count_big_freq_of_given_word.get(bigram[0], 0)
        # print("found one for ", token)
    
    # print(count_big_freq_of_given_word)
    with open('count_big_freq_of_given_word.pkl', 'wb') as f:
        pickle.dump(count_big_freq_of_given_word, f)

    


#bigram(Linear iterpolation smoothing)
#unigram_percenrage: a hash table that contain all p(w)
#Given_word, w_i-1
#lamda = 0.9
#DF_count, document freq
def LTS(unigram_percentage, bigram_freq_dict,query_word, given_word, tokens_count):
    bigram = (given_word,query_word)
    #c(w_i-1, w_i)
    if bigram in bigram_freq_dict:
        bigram_freq = bigram_freq_dict[bigram]
    else:
        bigram_freq = 0
    #c(w_i)
    given_word_freq = tokens_count[given_word]
    #(1-lamda)* c(w_i-1*w_i)/c(w_i-1)
    mle = (bigram_freq/given_word_freq) * 0.1
    #lamda*p(w_i)
    parameter = 0.9 * unigram_percentage[query_word]
    return mle + parameter




def LTS_next_word(unigram_percentage, bigram_freq_dict, given_word, tokens_count):
    words = []
    prob = []
    for token in tokens_count:
        query_word = token
        LTS_value = LTS(unigram_percentage, bigram_freq_dict, token, given_word, tokens_count)
        words.append(query_word)
        prob.append(LTS_value)
    prob = np.array(prob)
    prob /= prob.sum()
    random_index = np.random.choice(len(words), p=prob)
    return (words[random_index], prob[random_index])


def top_ten_LTS(unigram_percentage, bigram_freq_dict, given_word, tokens_count):
    total_LTS = 0
    top_ten_LTS_tokens = []  
    for word in tokens_count:
        token = word
        LTS_value = LTS(unigram_percentage, bigram_freq_dict, token, given_word, tokens_count)
        total_LTS += LTS_value
        # If the length of top_ten_LTS_tokens is less than 10, simply add the token
        if len(top_ten_LTS_tokens) < 10:
            heapq.heappush(top_ten_LTS_tokens, (LTS_value, token))
        else:
            # If the current LTS value is greater than the smallest LTS value in top_ten_LTS_tokens, replace it
            if LTS_value > top_ten_LTS_tokens[0][0]:
                heapq.heappop(top_ten_LTS_tokens)  # Remove the smallest LTS value
                heapq.heappush(top_ten_LTS_tokens, (LTS_value, token))  # Push the new LTS value and token

    # Once the loop is done, the top_ten_LTS_tokens list will contain the top 10 tokens with the highest LTS values
    top_ten_LTS_tokens = [(value, token) for value, token in sorted(top_ten_LTS_tokens, reverse=True)]
    
    
    with open ("LTS_top10.txt", "w") as f:
        print(top_ten_LTS_tokens, file=f)
        print("total prob" , total_LTS)
        
    

#bigram(aboslute dicounts smoothing)
#unigram_percenrage: a hash table that contain all p(w)
#Given_word, w_i-1
#DF_count, document freq
#unique_given_word_count: the count of how many unique word w_i-1(given_word), w_i will have 
def ADS(unigram_percentage, bigram_freq_dict, count_given_word_given_query_word, count_big_freq_of_given_word, query_word, given_word):
    bigram = (given_word,query_word)
    if bigram in bigram_freq_dict:
        #max(c(w_i,w_i-1),0)
        bigram_freq = max(bigram_freq_dict[bigram] - 0.1,0)
    else:
        bigram_freq = 0
    if query_word in count_given_word_given_query_word:
        #number of w_i-1 given the w_i:
        unique_count = count_given_word_given_query_word[query_word] * 0.1
    else:
        unique_count = 0
    #unigram prob of given_word
    given_word_freq = unigram_percentage[given_word]
    #number of bigram starting with w_i-1
    given_word_count = count_big_freq_of_given_word[given_word]
    return (bigram_freq + (unique_count * given_word_freq)) / given_word_count


def ADS_next_word(unigram_percentage, bigram_freq_dict, count_given_word_given_query_word, count_big_freq_of_given_word, tokens_count, given_word):
    words = []
    prob = []
    for token in tokens_count:
        query_word = token
        ADS_value = ADS(unigram_percentage, bigram_freq_dict, count_given_word_given_query_word, count_big_freq_of_given_word, token, given_word)
        words.append(query_word)
        prob.append(ADS_value)
    prob = np.array(prob)
    prob /= prob.sum()
    random_index = np.random.choice(len(words), p=prob)
    return (words[random_index], prob[random_index])

def top_ten_ADS(unigram_percentage, bigram_freq_dict, count_given_word_given_query_word, count_big_freq_of_given_word, given_word, tokens_count):
    top_ten_ADS_tokens = []  
    total_ADS = 0
    for word in tokens_count:
        token = word
        ADS_value = ADS(unigram_percentage, bigram_freq_dict, count_given_word_given_query_word, count_big_freq_of_given_word, token, given_word)
        total_ADS += ADS_value
        # If the length of top_ten_LTS_tokens is less than 10, simply add the token
        if len(top_ten_ADS_tokens) < 10:
            heapq.heappush(top_ten_ADS_tokens, (ADS_value, token))
        else:
            # If the current LTS value is greater than the smallest LTS value in top_ten_LTS_tokens, replace it
            if ADS_value > top_ten_ADS_tokens[0][0]:
                heapq.heappop(top_ten_ADS_tokens)  # Remove the smallest LTS value
                heapq.heappush(top_ten_ADS_tokens, (ADS_value, token))  # Push the new LTS value and token

    # Once the loop is done, the top_ten_ADS_tokens list will contain the top 10 tokens with the highest LTS values
    top_ten_ADS_tokens = [(value, token) for value, token in sorted(top_ten_ADS_tokens, reverse=True)]
    with open ("ADF_top10.txt", "w") as f:
        print(top_ten_ADS_tokens, file=f)
        print("total prob",total_ADS)

def unigram_genrate_doc(unigram_percentage, doc_len, total_doc):
    unigram_docs = []
    words = list(unigram_percentage.keys())
    prob = list(unigram_percentage.values())   
    prob = np.array(prob)
    prob /= prob.sum()
    for i in range(total_doc):
        doc = ""
        likelihood = 0
        for j in range(doc_len):
            random_index = np.random.choice(len(words), p=prob)
            cur_word = words[random_index]
            cur_word += " "
            doc += cur_word
            if likelihood == 0:
                likelihood = prob[random_index]
            else:
                likelihood = likelihood*prob[random_index]
        doc_item = (doc , likelihood)
        unigram_docs.append(doc_item)
    return unigram_docs

def LTS_genrate_doc(unigram_percentage, bigram_freq_dict, tokens_count, doc_len, total_doc):
    LTS_docs = []
    uni_words = list(unigram_percentage.keys())
    uni_prob = list(unigram_percentage.values())   
    uni_prob = np.array(uni_prob)
    uni_prob /= uni_prob.sum()
    for i in range(total_doc):
        start_word = uni_words[np.random.choice(len(uni_words), p=uni_prob)]
        doc = [start_word]
        likelihood = 0
        for i in range(1, doc_len):
            given_word = doc[i-1]
            cur_word = LTS_next_word(unigram_percentage, bigram_freq_dict, given_word, tokens_count) 
            doc.append(cur_word[0])
            if likelihood == 0:
                likelihood = cur_word[1]
            else:
                likelihood = likelihood * cur_word[1]
        doc_item = (doc, likelihood)
        LTS_docs.append(doc_item)
    return LTS_docs

def ADS_generate_doc(unigram_percentage, bigram_freq_dict, count_given_word_given_query_word, count_big_freq_of_given_word, tokens_count, doc_len, total_doc):
    ADS_docs = []
    uni_words = list(unigram_percentage.keys())
    uni_prob = list(unigram_percentage.values())   
    uni_prob = np.array(uni_prob)
    uni_prob /= uni_prob.sum()
    for i in range(total_doc):
        start_word = uni_words[np.random.choice(len(uni_words), p=uni_prob)]
        doc = [start_word]
        likelihood = 0
        for i in range(1, doc_len):
            given_word = doc[i-1]
            cur_word = ADS_next_word(unigram_percentage, bigram_freq_dict, count_given_word_given_query_word, count_big_freq_of_given_word, tokens_count, given_word)
            doc.append(cur_word[0])
            if likelihood == 0:
                likelihood = cur_word[1]
            else:
                likelihood = likelihood * cur_word[1]
        doc_item = (doc, likelihood)
        ADS_docs.append(doc_item)
    return ADS_docs
def main():
    # process_documents()
    with open("tokens_count.pkl", "rb") as f:
        tokens_count = pickle.load(f)

    with open("unigram_percentage.pkl", "rb") as f:
        unigram_percentage = pickle.load(f)

    with open("bigram_freq_dict.pkl", "rb") as f:
        bigram_freq_dict = pickle.load(f)


    with open("count_gien_word_given_query_word.pkl", "rb") as f:
        count_gien_word_given_query_word = pickle.load(f)
    
    with open("count_big_freq_of_given_word.pkl", "rb") as f:
        count_big_freq_of_given_word = pickle.load(f)


    #find top 10 word follow the given_word

    given_word = "good"
    # Find the top 10 most frequent word given the word "good using LTS" 
    top_ten_LTS(unigram_percentage, bigram_freq_dict, given_word, tokens_count)
    # Find the top 10 most frequent word given the word "good using ADS"
    top_ten_ADS(unigram_percentage, bigram_freq_dict, count_gien_word_given_query_word, count_big_freq_of_given_word, given_word, tokens_count)


    # generate docs 
    doc_len = 20
    total_doc = 10
    #unigram: 
    print("===============================UNI===============================")
    uni_doc = unigram_genrate_doc(unigram_percentage, doc_len, total_doc)
    with open("UNI_generate.txt", "w") as f:
        for doc in uni_doc:
            print(f"docs: {doc[0]}, doc's len: {len(doc[0].split())} likelihood: {doc[1]}", file=f)
    
    #LTS    
    print("===============================LTS===============================")
    LTS_docs = LTS_genrate_doc(unigram_percentage, bigram_freq_dict, tokens_count, doc_len, total_doc)
    with open("LTS_generate.txt", "w") as f:
        for doc in LTS_docs:
            sentence = ' '.join(doc[0])
            print(f"docs: {sentence}, doc's len: {len(doc[0])} likelihood: {doc[1]}", file=f)
    #ADS
    print("===============================ADS===============================")
    ADS_docs = ADS_generate_doc(unigram_percentage, bigram_freq_dict, count_gien_word_given_query_word, count_big_freq_of_given_word, tokens_count, doc_len, total_doc)
    with open("ADS_generate.txt", "w") as f:
        for doc in ADS_docs:
            sentence = ' '.join(doc[0])
            print(f"docs: {sentence}, doc's len: {len(doc[0])} likelihood: {doc[1]}", file=f)
    


            
    
    
    
if __name__ == "__main__":
    main()