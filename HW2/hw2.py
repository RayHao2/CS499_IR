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
        count_big_freq_of_given_word[bigram[0]] = bigram_frquency_dic[bigram] + count_big_freq_of_given_word.get(bigram[1], 0)
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
    if bigram in bigram_freq_dict:
        bigram_freq = bigram_freq_dict[bigram]
    else:
        bigram_freq = 0
    #c(w_i-1)
    given_word_freq = tokens_count[given_word]
    #(1-lamda)* c(w_i-1*w_i)/c(w_i-1)
    mle = (bigram_freq/given_word_freq) * 0.1
    #lamda*p(w_i)
    parameter = 0.9 * unigram_percentage[query_word]
    return mle + parameter




def LTS_next_word(unigram_percentage, bigram_freq_dict, given_word, tokens_count):
    value = float("-inf")
    output = ""
    for bigram in bigram_freq_dict:
        if bigram[0] == given_word:
            token = bigram[1]
            LTS_value = LTS(unigram_percentage, bigram_freq_dict, token, given_word, tokens_count)
            if value < LTS_value:
                output = token
    return output


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
def ADS(unigram_percentage, bigram_freq_dict, count_gien_word_given_query_word, count_big_freq_of_given_word, query_word, given_word, tokens_count):
    bigram = (given_word,query_word)
    if bigram in bigram_freq_dict:
        #max(c(w_i,w_i-1),0)
        bigram_freq = max(bigram_freq_dict[bigram] - 0.1,0)
    else:
        bigram_freq = 0

    if query_word in count_gien_word_given_query_word:
        #number of w_i-1 given the w_i:
        unique_count = count_gien_word_given_query_word[query_word] * 0.1
    else:
        unique_count = 0.1
    #unigram prob of given_word
    given_word_freq = unigram_percentage[given_word]
    #number of bigram starting with w_i-1
    given_word_count = count_big_freq_of_given_word[given_word]



    return (bigram_freq + (unique_count * given_word_freq)) / given_word_count


def top_ten_ADS(unigram_percentage, bigram_freq_dict, count_gien_word_given_query_word, count_big_freq_of_given_word, given_word, tokens_count):
    top_ten_ADS_tokens = []  
    total_ADS = 0
    for word in tokens_count:
        token = word
        ADS_value = ADS(unigram_percentage, bigram_freq_dict, count_gien_word_given_query_word, count_big_freq_of_given_word, token, given_word, tokens_count)
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
    for i in range(total_doc):
        doc = ""
        for j in range(doc_len):
            cur_word = random.choice(list(unigram_percentage.keys()))
            cur_word += " "
            doc += cur_word
        unigram_docs.append(doc)
    return unigram_docs

def LTS_genrate_doc(unigram_percentage, doc_len, total_doc, bigram_freq_dict, tokens_count):
    LTS_docs = []
    for i in range(total_doc):
        cur_word = random.choice(list(unigram_percentage.keys()))
        doc = []
        while len(doc) < 20:
            given_word = cur_word
            cur_word = LTS_next_word(unigram_percentage, bigram_freq_dict, given_word, tokens_count) 
            doc.append(cur_word)
        LTS_docs.append(doc)
    return LTS_docs
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


    #Find all bigram that came with good at first(W_i-1)
    given_word = "good"
    #Find the top 10 most frequent word given the word "good using LTS" 
    # top_ten_LTS(unigram_percentage, bigram_freq_dict, given_word, tokens_count)
    #Find the top 10 most frequent word given the word "good using ADS"
    top_ten_ADS(unigram_percentage, bigram_freq_dict, count_gien_word_given_query_word, count_big_freq_of_given_word, given_word, tokens_count)



    # sum = 0
    # for word in tokens_count:
    #     sum += LTS(unigram_percentage, bigram_freq_dict, word, given_word, tokens_count)
    # print(sum)
    #generate docs
    # doc_len = 20
    # total_doc = 10
    # print("===============================UNI===============================")
    # #unigram: 
    # unigram_docs = unigram_genrate_doc(unigram_percentage, doc_len, total_doc)
    # for doc in unigram_docs:
    #     print(doc)
    # print("===============================LTS===============================")
    # #LTS
    # LTS_docs = LTS_genrate_doc(unigram_percentage, doc_len, total_doc, bigram_freq_dict, tokens_count)
    # for doc in LTS_docs:
    #     sentence = ' '.join(doc)
    #     print(sentence)
    #ADS
    

    


            
    
    
    
if __name__ == "__main__":
    main()