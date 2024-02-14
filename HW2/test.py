from nltk.util import bigrams
from nltk.tokenize import word_tokenize
import os
import json
import time
current_dir = os.getcwd()
path = os.path.join(os.path.dirname(current_dir), "yelp")
yelp_store_list = os.listdir(path)
given_word = "good"

freq = 0
for yelp_store in yelp_store_list:
    store_path = path + "/" + yelp_store
    with open(store_path, 'r') as file:
        data = json.load(file)
    reviews = data["Reviews"]
    for review in reviews:
        doc = review['Content']
        tokens = word_tokenize(doc)
        bigram_list = list(bigrams(tokens))
        for bigram in bigram_list:
            if given_word in bigram:
                print(bigram)