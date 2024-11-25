'''
@author: Sougata Saha
@modifier: Divyesh Pratap Singh
Institute: University at Buffalo
'''

import collections
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()
        
    def get_doc_id(self, doc):
        """ Splits each line of the document, into doc_id & text.
            Already implemented"""
        arr = doc.split("\t")     
        return int(arr[0]), arr[1]

    def get_doc_id_and_text(self, json_item):
        """
        Extract doc_id (revision_id) and document text from a JSON item.
        """
        revision_id = json_item.get('revision_id')
        title = json_item.get('title', '')
        summary = json_item.get('summary', '')
        combined_text = f"{title} {summary}"
        return revision_id, combined_text

    def tokenizer(self, text):
        """ Implement logic to pre-process & tokenize document text.
            Write the code in such a way that it can be re-used for processing the user's query.
            To be implemented."""
        sentence = text.lower()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()  
        words = sentence.split()
        words = [w for w in words if not w in self.stop_words]
        tokens = [self.ps.stem(token) for token in words]
        return tokens
