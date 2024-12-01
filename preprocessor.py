'''
@author: Sougata Saha
@modifier: Divyesh Pratap Singh, Dakshesh Gusain
Institute: University at Buffalo
'''

import collections
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def get_doc_id(self, doc):
        arr = doc.split("\t")     
        return int(arr[0]), arr[1]

    def get_doc_id_and_text_and_topic_and_title(self, json_item):
        revision_id = json_item.get('revision_id')
        topic = json_item.get('topic', '')
        title = json_item.get('title', '')
        summary = json_item.get('summary', '')
        combined_text = f"{title} {summary}"
        return revision_id, combined_text, topic, title

    def tokenizer(self, text):
        sentence = text.lower()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()  
        words = sentence.split()
        words = [w for w in words if not w in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in words]
        return tokens
