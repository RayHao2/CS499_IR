import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import sys



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
    #calulate unigram percentage
    for token in tokens_count:
        print(token)
        count = tokens_count[token]
        unigram_percentage[token] = count / total_tokens

    
    #bigram(Linear iterpolation smoothing)
    
    #bigram(aboslute dicounts smoothing)
def main():
    part_one()
    
    
    
if __name__ == "__main__":
    main()