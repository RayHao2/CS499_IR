import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import sys
import pickle



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
    return unigram_percentage, DF_count, total_tokens



#return the number of bigram where it fits the format of (given_word, query_word) assuming the word already done the normalization process
def bigram_frquency(query_word,given_word):
    freq = 0
    
    return freq


#bigram(Linear iterpolation smoothing)
#unigram_percenrage: a hash table that contain all p(w)
#Given_word, w_i-1
#DF_count, document freq
def LTS(unigram_percentage, query_word, given_word, tokens_count):

    #c(w_i-1)
    given_word_freq = unigram_percentage[given_word]
    #c(w_i-1*w_i)
    bigram_freq = bigram_frquency(query_word, given_word)
    #(1-lamda)* c(w_i-1*w_i)/c(w_i-1)
    mle = 0.1 * (bigram_freq/given_word_freq)
    parameter = 0.9*(given_word_freq/tokens_count)
    return mle + parameter

#bigram(aboslute dicounts smoothing)
def ADS(unigram_percentage, given_word):
    pass
def main():
    with open("total_count.txt", "r") as f:
        tokens_count = int(f.read())
    with open("unigram_percentage.pkl", "rb") as f:
        unigram_percentage = pickle.load(f)
    print(unigram_percentage)
    
    
    
if __name__ == "__main__":
    main()