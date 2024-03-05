import numpy as np

# Sample numpy array with tuples
def cos_sim(query_vec, doc_vec):
    return np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))


q = [0.1234]
d = [0.23131452]


