import multiprocessing
from gensim.models import Word2Vec
import gensim.downloader
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import json
from gensim.test.utils import common_texts
import pickle
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
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
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    documents = []
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

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
                elif token in stop_words:
                    continue
                elif is_integer(token):
                    token = "NUM"
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                normaled_token.append(token)
            documents.append(normaled_token)
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents,f)

def process_query():
    init_query= [
        "general chicken", "fried chicken", "BBQ sandwiches", "mashed potatoes", "Grilled Shrimp Salad",
        "lamb Shank", "Pepperoni pizza", "brussel sprout salad", "FRIENDLY STAFF", "Grilled Cheese"
    ]
    stemmer = PorterStemmer()
    procssed_query = []
    for query in init_query:
        normal_query = []
        query = word_tokenize(query)
        for token in query:
            token = stemmer.stem(token)
            token = token.lower()
            normal_query.append(token)
        procssed_query.append(normal_query)
    with open("query.pkl", "wb") as f:
        pickle.dump(procssed_query,f)
        

def word2Vec():
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    with open("query.pkl", "wb") as f:
        querys = pickle.load(f)
    # with open("outpu.txt", "w") as f:
    #     for doc in documents:
    #         print(doc,file=f)
    
    cores = multiprocessing.cpu_count() 

    w2v_model = Word2Vec(min_count=20,
                        window=2,
                        size=300,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)

    
    


def main(): 
    process_query()
    # process_documents()
    # word2Vec()
    


if __name__ == "__main__":
    main()