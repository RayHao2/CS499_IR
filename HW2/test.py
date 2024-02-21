from nltk.util import bigrams
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import json
import time
import re
import numpy as np
def remove_non_alphanumeric(input_string):
    # Use regular expression to keep only English letters and numbers
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    


words = ["HI", "My", "Name", "IS"]
prob = [0,0,0,0.9]
prob = np.array(prob)
prob /= prob.sum()
random_index = np.random.choice(len(words), p=prob)
print(random_index)

