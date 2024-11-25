'''
@author: Sougata Saha
@modifier: Divyesh Pratap Singh
Institute: University at Buffalo
'''

from linkedlist import LinkedList
from collections import OrderedDict


class Indexer:
    def __init__(self):
        """ Add more attributes if needed"""
        self.inverted_index = OrderedDict({})
        self.total_documents = 0
        self.document_frequency = {}
        self.token_count = {}

    def get_index(self):
        """ Function to get the index.
            Already implemented."""
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """ This function adds each tokenized document to the index. This in turn uses the function add_to_index
            Already implemented."""
        
        self.total_documents += 1
        unique_terms = set(tokenized_document)

        for t in tokenized_document:
            self.add_to_index(t, doc_id)

        for term in unique_terms:
            if term in self.document_frequency:
                self.document_frequency[term] += 1
            else:
                self.document_frequency[term] = 1

        self.token_count[doc_id] = len(tokenized_document)

    def add_to_index(self, term_, doc_id_):
        """ This function adds each term & document id to the index.
            If a term is not present in the index, then add the term to the index & initialize a new postings list (linked list).
            If a term is present, then add the document to the appropriate position in the posstings list of the term.
            To be implemented."""
        
        if term_ not in self.inverted_index:
            self.inverted_index[term_] = LinkedList()
        self.inverted_index[term_].insert_in_order(doc_id_)

    def sort_terms(self):
        """ Sorting the index by terms.
            Already implemented."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        """ For each postings list in the index, add skip pointers.
            To be implemented."""
        for v in self.inverted_index.values():
            v.add_skip_connections()
        return

    def calculate_tf_idf(self):
        """ Calculate tf-idf score for each document in the postings lists of the index.
            To be implemented."""
        
        for term, posting_list in self.inverted_index.items():
            idf = self.total_documents/ self.document_frequency.get(term, 1)
            current = posting_list.start_node
            while current:
                doc_id = current.value
                term_frequency = current.tf / self.token_count[doc_id]
                current.tfidf = term_frequency * idf
                current = current.next