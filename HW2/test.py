from nltk.util import bigrams
from nltk.tokenize import word_tokenize
import os
import json
import time
import re
def remove_non_alphanumeric(input_string):
    # Use regular expression to keep only English letters and numbers
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result


input = "."
print(remove_non_alphanumeric(input))
print("end")