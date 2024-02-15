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



def is_integer(s):
    try:
        int(s)
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


def part_one():
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    tokens_count = {}
    unigram_percentage = {}
    DF_count = {}
    stemmer = PorterStemmer()
    total_tokens = 0
    yelp_store_list = os.listdir(path)
    
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
            doc_ID = review['ReviewID']
            tokens = word_tokenize(doc) #token the review content
            #Loop through each token and normal it
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif is_integer(token):
                    token = "NUM"
                    print("NUM appear")
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                total_tokens += 1
                tokens_count[token] = 1 + tokens_count.get(token, 0)
                if token in DF_count and DF_count[token] is not None:
                    DF_count[token].add(doc_ID)
                else:
                    DF_count[token] = set()
                    DF_count[token].add(doc_ID)

    #calulate unigram percentage
    for token in tokens_count:
        print(token)
        count = tokens_count[token]
        unigram_percentage[token] = count / total_tokens

    #calulate DFF
    for key in DF_count:
        DF_count[key] = len(DF_count[key])
    
    with open("output.txt", "w") as f:
        sys.stdout = f
        print(tokens_count)
        
    with open("tokens_count.pkl", "wb") as f:
        pickle.dump(tokens_count,f)
    with open("unigram_percentage.pkl", "wb") as f:
        pickle.dump(unigram_percentage,f)
    with open("total_tokens.txt", "w") as f:
        f.write(str(total_tokens))




#return the number of bigram where it fits the format of (given_word, query_word) assuming the word already done the normalization process
def bigram_frquency(given_word):
    bigram_freq_dict = {}
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    yelp_store_list = os.listdir(path)
    stemmer = PorterStemmer()
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        #Loop through each review in the yelp _ file
        for review in reviews:
            doc = review['Content']
            tokens = word_tokenize(doc)
            normaled_token = []
            #normal token
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif is_integer(token):
                    token = "NUM"
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                normaled_token.append(token)
            #find bigram in the normaled token
            bigram_list = list(bigrams(normaled_token))
            for bigram in bigram_list:
                if given_word == bigram[0]:
                    bigram_freq_dict[bigram[1]] = 1 + bigram_freq_dict.get(bigram[1],0)
    return bigram_freq_dict


#bigram(Linear iterpolation smoothing)
#unigram_percenrage: a hash table that contain all p(w)
#Given_word, w_i-1
#lamda = 0.9
#DF_count, document freq
def LTS(unigram_percentage, bigram_freq_dict,query_word, given_word, tokens_count):
    if given_word in tokens_count and query_word in bigram_freq_dict and query_word in unigram_percentage:
        #c(w_i-1)
        given_word_freq = tokens_count[given_word]
        #c(w_i-1,w_i)
        bigram_freq = bigram_freq_dict[query_word]
        #(1-lamda)* c(w_i-1*w_i)/c(w_i-1)
        mle = 0.1 * (bigram_freq/given_word_freq)
        parameter = 0.9 * unigram_percentage[query_word]
        return mle + parameter
    else:
        return 0

#bigram(aboslute dicounts smoothing)
#unigram_percenrage: a hash table that contain all p(w)
#Given_word, w_i-1
#DF_count, document freq
def ADS(unigram_percentage, query_word, given_word, tokens_count):
    pass
def main():
    with open("tokens_count.pkl", "rb") as f:
        tokens_count = pickle.load(f)
    with open("total_count.txt", "r") as f:
        total_count = int(f.read())
    with open("unigram_percentage.pkl", "rb") as f:
        unigram_percentage = pickle.load(f)


    # #find the top 10 word corresponding to LTS
    given_word = "good"
    bigram_freq_dict = bigram_frquency(given_word)

    #Finding the top 10 most frequent word given the word "good"
    top_ten_LTS_tokens = []  
    for token in tokens_count:
        LTS_value = LTS(unigram_percentage, bigram_freq_dict,token, given_word, tokens_count)
        print(f"Token: {token},LTS:{LTS_value}")
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
    with open("output.txt", "w") as f:
        sys.stdout = f
        print(top_ten_LTS_tokens)
  
            
    
    
    
if __name__ == "__main__":
    main()