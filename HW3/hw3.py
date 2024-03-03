import multiprocessing
import time
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
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

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
        
def average_vector(doc, model):
    output = []
    for word in doc:
        cur_term_avg_weight = np.mean(model.wv[word])
        output.append(cur_term_avg_weight)
    return output



def cos_sim(query_vec, doc_vec):
    if len(query_vec) < len(doc_vec):
        query_vec_padding = np.pad(query_vec, (0, len(doc_vec) - len(query_vec)), mode='constant')
        return np.dot(query_vec_padding, doc_vec) / (np.linalg.norm(query_vec_padding) * np.linalg.norm(doc_vec))
    
    elif len(query_vec) > len(doc_vec):
        doc_vec_padding = np.pad(doc_vec, (0, len(query_vec) - len(doc_vec)), mode='constant')
        return np.dot(query_vec, doc_vec_padding) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec_padding))
    
    
    else:
        return np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        

        
def word2Vec():
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    with open("query.pkl", "rb") as f:
        querys = pickle.load(f)
    # with open("output.txt", "w") as f:
        # for doc in documents:
        #     print(doc,file=f)
    
    train_doc = documents + querys
    
    #uncomment if first run
    # start_time = time.time()
    # cores = multiprocessing.cpu_count() 
    # model = Word2Vec(sentences=train_doc, vector_size=100, window=5, min_count=1, workers=cores-1)
    # model.save("word2vec.model")
    # end_time = time.time()
    # print("Train time: ", end_time - start_time)
    
    
    #load the model
    model = Word2Vec.load("word2vec.model")
    doc_vectors = []
    query_vectors = []
    

    
    for doc in documents:
        doc_vectors.append(average_vector(doc,model))
    for query in querys:
        query_vectors.append(average_vector(query,model))

    print(cos_sim(query_vectors[0], doc_vectors[0]))
    
    with open("output.txt", "w") as f:
        f.write("")
    
    #calulate cosine sim   
    for i in range(len(query_vectors)):
        query_vec = query_vectors[i]
        top_similarities = []
        for j in range(len(doc_vectors)):
            sim = cos_sim(query_vec,doc_vectors[j])
            if len(top_similarities) < 3:
                heapq.heappush(top_similarities, (sim, j))
            else:
                if sim > top_similarities[0][0]:
                    heapq.heappop(top_similarities)  
                    heapq.heappush(top_similarities, (sim, j))

        with open("output.txt", "a") as f:
            print(f"Query: {querys[i]}", file=f)
        for sim in top_similarities:
            with open("output.txt", "a") as f:
                print(f"{sim[0]} {documents[sim[1]]}", file=f)
            
   
def tfidf():
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    with open("query.pkl", "rb") as f:
        querys = pickle.load(f)
    
    train_doc = documents + querys
    dct = Dictionary(train_doc)
    train_doc_corpus = [dct.doc2bow(doc) for doc in train_doc] 
    
    #run first time
    start_time = time.time()
    model = TfidfModel(train_doc_corpus)
    end_time = time.time()
    print("Train time: ", end_time - start_time)
    model.save("tfidf.model")


    #convert documents and querys into bow
    dct = Dictionary(documents)
    doc_bow = [dct.doc2bow(doc) for doc in documents] 
    dct = Dictionary(querys)
    query_bow = [dct.doc2bow(doc) for doc in querys] 
    
    #get the vector for querys and documents
    doc_vectors = []
    query_vectors = []
    for doc in doc_bow:
        vec = model[doc]
        new_vec = [x[1] for x in vec]
        doc_vectors.append(new_vec)
    for query in query_bow:
        vec = model[query]
        new_vec = [x[1] for x in vec]
        query_vectors.append(new_vec)


    #calulate cosine sim   
    for i in range(len(query_vectors)):
        query_vec = query_vectors[i]
        top_similarities = []
        for j in range(len(doc_vectors)):
            sim = np.dot(query_vec,doc_vectors[j]) / (norm(query_vec) * norm(doc_vectors[j]))
            if len(top_similarities) < 3:
                heapq.heappush(top_similarities, (sim, j))
            else:
                if sim > top_similarities[0][0]:
                    heapq.heappop(top_similarities)  
                    heapq.heappush(top_similarities, (sim, j))

        with open("output.txt", "a") as f:
            print(f"Query: {querys[i]}", file=f)
        for sim in top_similarities:
            with open("output.txt", "a") as f:
                print(f"{sim[0]} {documents[sim[1]]}", file=f)
    

    
    # print(vector)

def main(): 
    # process_query()
    # process_documents()
    word2Vec()
    # tfidf()
    


if __name__ == "__main__":
    main()