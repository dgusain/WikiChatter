# indexing_pipeline.py

import json
import math
from tqdm import tqdm
from preprocessor import Preprocessor
from indexer import Indexer

class IndexingPipeline:
    def __init__(self, corpus_path, index_path, metadata_path):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    def run_indexer(self):
        """Read the corpus and build the inverted index."""
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
        except FileNotFoundError:
            print(f"Error: The file '{self.corpus_path}' was not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: The file '{self.corpus_path}' does not contain valid JSON.")
            return

        print("Indexing documents...")
        for item in tqdm(data, total=len(data)):
            doc_id, document = self.preprocessor.get_doc_id_and_text(item)
            if not doc_id or not document:
                continue  # Skip if essential fields are missing
            tokenized_document = self.preprocessor.tokenizer(document)
            self.indexer.generate_inverted_index(doc_id, tokenized_document)

        print("Sorting terms in the inverted index...")
        self.indexer.sort_terms()

        print("Adding skip connections to postings lists...")
        self.indexer.add_skip_connections()

        print("Calculating TF-IDF scores...")
        self.indexer.calculate_tf_idf()

        print("Indexing completed.")

        # Convert the inverted index to a JSON-serializable format
        print("Converting inverted index to JSON format...")
        json_inverted_index = {}
        for term, postings_list in self.indexer.get_index().items():
            json_inverted_index[term] = postings_list.get_postings()

        # Persist the inverted index as JSON
        print(f"Saving inverted index to '{self.index_path}'...")
        with open(self.index_path, 'w', encoding='utf-8') as idx_file:
            json.dump(json_inverted_index, idx_file, indent=4)

        # Save metadata as JSON
        metadata = {
            'total_documents': self.indexer.total_documents,
            'document_frequency': self.indexer.document_frequency,
            'token_count': self.indexer.token_count
        }
        print(f"Saving metadata to '{self.metadata_path}'...")
        with open(self.metadata_path, 'w', encoding='utf-8') as meta_file:
            json.dump(metadata, meta_file, indent=4)

        print("All data has been successfully saved.")

if __name__ == "__main__":
    CORPUS_PATH = 'final_scrapped.json'
    INDEX_PATH = 'inverted_index.json'
    METADATA_PATH = 'metadata.json'

    pipeline = IndexingPipeline(CORPUS_PATH, INDEX_PATH, METADATA_PATH)
    pipeline.run_indexer()
