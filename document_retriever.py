# document_retriever.py

import json
import math
import re
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def clean_text(self, text):
        """
        Lowercase, remove punctuation, and non-alphanumeric characters.
        """
        text = text.lower()
        text = text.translate(self.punctuation_table)
        text = re.sub(r'\d+', '', text)
        return text

    def tokenize(self, text):
        """
        Tokenize text, remove stopwords, and perform stemming.
        """
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return tokens

class DocumentRetriever:
    def __init__(self, inverted_index_path, metadata_path, scrapped_data_path):
        self.preprocessor = Preprocessor()
        self.inverted_index = self.load_inverted_index(inverted_index_path)
        self.metadata = self.load_metadata(metadata_path)
        self.doc_id_to_summary = self.load_scrapped_data(scrapped_data_path)

    def load_inverted_index(self, path):
        """
        Load the inverted index from a JSON file.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                inverted_index = json.load(f)
            print(f"Inverted index loaded from '{path}'.")
            return inverted_index
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def load_metadata(self, path):
        """
        Load metadata from a JSON file.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"Metadata loaded from '{path}'.")
            return metadata
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def load_scrapped_data(self, path):
        """
        Load the scrapped data and create a mapping from doc_id to summary.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            doc_id_to_summary = {}
            for item in data:
                revision_id = item.get('revision_id')
                summary = item.get('summary', '')
                if revision_id:
                    doc_id_to_summary[revision_id] = summary
            print(f"Scrapped data loaded from '{path}'.")
            return doc_id_to_summary
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def _daat_and_with_tfidf(self, terms):
        """
        Document-at-a-Time AND retrieval with TF-IDF scoring.
        Returns the top-k documents sorted by cumulative TF-IDF scores.
        """
        inverted_index = self.inverted_index
        metadata = self.metadata
        total_documents = metadata.get('total_documents', 0)

        # Initialize postings lists for each term
        postings_lists = []
        for term in terms:
            if term in inverted_index:
                postings = inverted_index[term]
                postings_lists.append(set([posting['doc_id'] for posting in postings]))
            else:
                # If any term is not in the index, intersection is empty
                return [], 0

        # Intersection of all postings lists
        intersection = set.intersection(*postings_lists) if postings_lists else set()
        total_comparisons = sum(len(p) for p in postings_lists)

        if not intersection:
            return [], total_comparisons

        # Calculate cumulative TF-IDF scores
        doc_tfidf_scores = defaultdict(float)
        for doc_id in intersection:
            for term in terms:
                postings = inverted_index.get(term, [])
                for posting in postings:
                    if posting['doc_id'] == doc_id:
                        tf = posting['tfidf']
                        doc_tfidf_scores[doc_id] += tf
                        break  # Move to next term once found

        # Sort documents by cumulative TF-IDF scores in descending order
        sorted_docs = sorted(doc_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs, total_comparisons

    def retrieve_top_k(self, query, k=10):
        """
        Given a query, retrieve the top-k relevant document summaries.
        """
        # Preprocess the query
        tokens = self.preprocessor.tokenize(query)
        if not tokens:
            print("No valid terms found in the query after preprocessing.")
            return []

        print(f"Processed Query Terms: {tokens}")

        # Retrieve and rank documents
        ranked_docs, comparisons = self._daat_and_with_tfidf(tokens)
        print(f"Total Comparisons Made: {comparisons}")
        print(f"Number of Documents Retrieved: {len(ranked_docs)}")

        # Get top-k documents
        top_k_docs = ranked_docs[:k]
        print(f"Top {k} Documents:")
        for rank, (doc_id, score) in enumerate(top_k_docs, start=1):
            print(f"{rank}. Doc ID: {doc_id}, TF-IDF Score: {score:.4f}")

        # Fetch summaries
        summaries = []
        for doc_id, score in top_k_docs:
            summary = self.doc_id_to_summary.get(doc_id, "No summary available.")
            summaries.append({
                'doc_id': doc_id,
                'summary': summary,
                'score': score
            })

        return summaries

def main():
    # Paths to the necessary files
    INVERTED_INDEX_PATH = 'inverted_index.json'
    METADATA_PATH = 'metadata.json'
    SCRAPPED_DATA_PATH = 'final_scrapped.json'

    # Initialize the DocumentRetriever
    retriever = DocumentRetriever(INVERTED_INDEX_PATH, METADATA_PATH, SCRAPPED_DATA_PATH)

    # Sample queries for testing
    sample_queries = [
        "What is COVID?",
    ]

    for query in sample_queries:
        print("\n" + "="*50)
        print(f"Query: {query}")
        summaries = retriever.retrieve_top_k(query, k=10)
        if summaries:
            print("\nTop-K Summaries:")
            for idx, doc in enumerate(summaries, start=1):
                print(f"\nDocument {idx}:")
                print(f"Doc ID: {doc['doc_id']}")
                print(f"Score: {doc['score']:.4f}")
                print(f"Summary: {doc['summary']}")
        else:
            print("No relevant documents found for the query.")

if __name__ == "__main__":
    main()
