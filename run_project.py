'''
@author: Sougata Saha
@Modifier: Divyesh Pratap Singh
Institute: University at Buffalo
'''

from tqdm import tqdm
from preprocessor import Preprocessor
from indexer import Indexer
from collections import OrderedDict
from linkedlist import LinkedList
import inspect as inspector
import sys
import argparse
import json
import time
import random
import flask
from flask import Flask
from flask import request
import hashlib

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    def _merge(self, list1, list2):
        i, j = 0, 0
        intersection = []
        comparisons = 0

        while i < len(list1) and j < len(list2):
            comparisons += 1

            if list1[i] == list2[j]:
                intersection.append(list1[i])
                i += 1
                j += 1
            elif list1[i] < list2[j]:
                i += 1
            else:
                j += 1

        return intersection, comparisons

    def get_postings(self, term):
        
        tokenized_term = self.preprocessor.tokenizer(term)
        
        if not tokenized_term:
            print(f"The term '{term}' is maybe be a stopword.")
            return []
        
        processed_term = tokenized_term[0]
        
        if processed_term in self.indexer.inverted_index:
            postings_list = self.indexer.inverted_index[processed_term]
            postings = []
            current = postings_list.start_node
            while current:
                postings.append(current.value)
                current = current.next
            return postings
        else:
            print(f"The term '{term}' looks like not present in the inverted index.")
            return []

    def get_postings_skip_pointers(self, term):

        tokenized_term = self.preprocessor.tokenizer(term)
        
        if not tokenized_term:
            print(f"The term '{term}' is maybe a stopword.")
            return []
        
        processed_term = tokenized_term[0]
        
        if processed_term in self.indexer.inverted_index:
            postings_list = self.indexer.inverted_index[processed_term]
            postings = []
            current = postings_list.start_node
            while current.skip_pointer:
                postings.append(current.value)
                current = current.skip_pointer
                
            postings.append(current.value)
            return postings
        else:
            print(f"The term '{term}' looks like not present in the inverted index.")
            return []


    def _daat_and_without_skippointer(self, terms):
        """ Implement the DAAT AND algorithm, which merges the postings list of N query terms.
            Use appropriate parameters & return types.
            To be implemented."""
        
        postings_lists = []
        for term in terms:
            tokenized_terms = self.preprocessor.tokenizer(term)
            
            if not tokenized_terms:
                print(f"The term: '{term}' is maybe a stopword.")
                postings_lists.append([])
                continue

            processed_term = tokenized_terms[0]
            
            if processed_term in self.indexer.inverted_index:
                postings_list = self.indexer.inverted_index[processed_term]
                
                doc_ids = postings_list.traverse_list()
                postings_lists.append(doc_ids)
            else:
                print(f"The term: '{term}' looks like not present in the inverted index.")
                postings_lists.append([])

        if any(len(postinglist) == 0 for postinglist in postings_lists):
            return [], 0

        postings_lists.sort(key=lambda x: len(x))

        intersection = postings_lists[0]
        total_comparisons = 0

        for i in range(1, len(postings_lists)):
            intersection, comparisons = self._merge(intersection, postings_lists[i])
            total_comparisons += comparisons

            if not intersection:
                break

        return intersection, total_comparisons

    def _advance_pointer(self, pointer, target):

        comparisons = 0
        while pointer and pointer.value < target:
            comparisons += 1
            if pointer.skip_pointer and pointer.skip_pointer.value <= target:
                comparisons += 1
                pointer = pointer.skip_pointer
            else:
                pointer = pointer.next
                comparisons += 1
        return pointer, comparisons

    def _daat_and_with_skip_pointer(self, terms):

        processed_terms = []
        for term in terms:
            tokenized = self.preprocessor.tokenizer(term)
            if not tokenized:
                print(f"term: '{term}' is maybe a stopword.")
                continue
            processed_terms.append(tokenized[0])

        postings_lists = []
        for term in processed_terms:
            if term in self.indexer.inverted_index:
                postings_list = self.indexer.inverted_index[term]
                postings_lists.append(postings_list)
            else:
                print(f"The term: '{term}' looks like not present in the inverted index.")
                return [], 0

        postings_lists.sort(key=lambda postinglen: postinglen.length)

        pointers = [postinglen.start_node for postinglen in postings_lists]

        intersection = []
        total_comparisons = 0

        while True:
            if any(pointer is None for pointer in pointers):
                break

            current_doc_ids = [pointer.value for pointer in pointers]
            max_doc_id = max(current_doc_ids)

            for i in range(len(pointers)):
                ptr = pointers[i]
                if ptr.value < max_doc_id:
                    new_ptr, comparisons = self._advance_pointer(ptr, max_doc_id)
                    pointers[i] = new_ptr
                    total_comparisons += comparisons

            if any(pointer is None for pointer in pointers):
                break

            current_doc_ids = [pointer.value for pointer in pointers]
            if all(doc_id == max_doc_id for doc_id in current_doc_ids):
                intersection.append(max_doc_id)

                for i in range(len(pointers)):
                    pointers[i] = pointers[i].next
                    total_comparisons += 1
        total_comparisons = int(total_comparisons/2)
        return intersection, total_comparisons

    def _daat_and_with_tfidf(self, terms):

        intersection, total_comparisons = self._daat_and_without_skippointer(terms)

        if not intersection:
            return [], total_comparisons

        doc_tfidf_scores = {}
        for doc_id in intersection:
            cumulative_tfidf = 0.0
            for term in terms:
                tokenized_term = self.preprocessor.tokenizer(term)
                if not tokenized_term:
                    continue
                processed_term = tokenized_term[0]
                if processed_term in self.indexer.inverted_index:
                    postings_list = self.indexer.inverted_index[processed_term]
                    current = postings_list.start_node
                    while current and current.value < doc_id:
                        current = current.next
                    if current and current.value == doc_id:
                        cumulative_tfidf += current.tfidf
            doc_tfidf_scores[doc_id] = cumulative_tfidf

        sorted_results = sorted(intersection, key=lambda doc: doc_tfidf_scores[doc], reverse=True)

        return sorted_results, total_comparisons

    def _daat_and_with_skip_pointer_tfidf(self, terms):

        intersection, total_comparisons = self._daat_and_with_skip_pointer(terms)

        if not intersection:
            return [], total_comparisons

        doc_tfidf_scores = {}
        for doc_id in intersection:
            cumulative_tfidf = 0.0
            for term in terms:
                tokenized_term = self.preprocessor.tokenizer(term)
                if not tokenized_term:
                    continue
                processed_term = tokenized_term[0]
                if processed_term in self.indexer.inverted_index:
                    postings_list = self.indexer.inverted_index[processed_term]
                    current = postings_list.start_node
                    while current and current.value < doc_id:
                        current = current.next
                    if current and current.value == doc_id:
                        cumulative_tfidf += current.tfidf
            doc_tfidf_scores[doc_id] = cumulative_tfidf

        sorted_results = sorted(intersection, key=lambda doc: doc_tfidf_scores[doc], reverse=True)

        return sorted_results, total_comparisons

    def _output_formatter(self, op):
        """ This formats the result in the required format.
            Do NOT change."""
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus):
        """ This function reads & indexes the corpus. After creating the inverted index,
            it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
            Already implemented, but you can modify the orchestration, as you seem fit."""
        with open(corpus, 'r', encoding='utf-8') as fp:
            for line in tqdm(fp.readlines()):
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()

    def sanity_checker(self, command):
        """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """
        # Sanity checker is not to be used as per piazza post
        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].start_node),
                "node_type": str(type(index[kw].start_node)),
                "node_value": str(index[kw].start_node.value),
                "command_result": eval(command) if "." in command else ""}

    def run_queries(self, query_list):
        """ DO NOT CHANGE THE output_dict definition"""
        output_dict = {
            'postingsList': {},
            'postingsListSkip': {},
            'daatAnd': {},
            'daatAndSkip': {},
            'daatAndTfIdf': {},
            'daatAndSkipTfIdf': {}
        }

        for query in tqdm(query_list):
            """ Run each query against the index. You should do the following for each query:
                1. Pre-process & tokenize the query.
                2. For each query token, get the postings list & postings list with skip pointers.
                3. Get the DAAT AND query results & number of comparisons with & without skip pointers.
                4. Get the DAAT AND query results & number of comparisons with & without skip pointers, 
                    along with sorting by tf-idf scores."""


            input_term_arr = self.preprocessor.tokenizer(query)

            for term in input_term_arr:
                postings = self.get_postings(term)
                skip_postings = self.get_postings_skip_pointers(term)
                """ Implement logic to populate initialize the above variables.
                    The below code formats your result to the required format.
                    To be implemented."""

                output_dict['postingsList'][term] = postings
                output_dict['postingsListSkip'][term] = skip_postings

            and_op_no_skip, and_op_skip, and_op_no_skip_sorted, and_op_skip_sorted = None, None, None, None
            and_comparisons_no_skip, and_comparisons_skip, \
                and_comparisons_no_skip_sorted, and_comparisons_skip_sorted = None, None, None, None
            """ Implement logic to populate initialize the above variables.
                The below code formats your result to the required format.
                To be implemented."""
            and_op_no_score_no_skip, and_results_cnt_no_skip = self._output_formatter(and_op_no_skip)
            and_op_no_score_skip, and_results_cnt_skip = self._output_formatter(and_op_skip)
            and_op_no_score_no_skip_sorted, and_results_cnt_no_skip_sorted = self._output_formatter(and_op_no_skip_sorted)
            and_op_no_score_skip_sorted, and_results_cnt_skip_sorted = self._output_formatter(and_op_skip_sorted)
            
            daat_results, daat_comparisons = self._daat_and_without_skippointer(input_term_arr)

            daat_skip_results, daat_skip_comparisons = self._daat_and_with_skip_pointer(input_term_arr)

            daat_tfidf_results, daat_tfidf_comparisons = self._daat_and_with_tfidf(input_term_arr)

            daat_skip_tfidf_results, daat_skip_tfidf_comparisons = self._daat_and_with_skip_pointer_tfidf(input_term_arr)

            output_dict['daatAnd'][query.strip()] = {}
            output_dict['daatAnd'][query.strip()]['results'] = daat_results
            output_dict['daatAnd'][query.strip()]['num_docs'] = len(daat_results)
            output_dict['daatAnd'][query.strip()]['num_comparisons'] = daat_comparisons

            output_dict['daatAndSkip'][query.strip()] = {}
            output_dict['daatAndSkip'][query.strip()]['results'] = daat_skip_results
            output_dict['daatAndSkip'][query.strip()]['num_docs'] = len(daat_skip_results)
            output_dict['daatAndSkip'][query.strip()]['num_comparisons'] = daat_skip_comparisons

            output_dict['daatAndTfIdf'][query.strip()] = {}
            output_dict['daatAndTfIdf'][query.strip()]['results'] = daat_tfidf_results
            output_dict['daatAndTfIdf'][query.strip()]['num_docs'] = len(daat_tfidf_results)
            output_dict['daatAndTfIdf'][query.strip()]['num_comparisons'] = daat_tfidf_comparisons

            output_dict['daatAndSkipTfIdf'][query.strip()] = {}
            output_dict['daatAndSkipTfIdf'][query.strip()]['results'] = daat_skip_tfidf_results
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_docs'] = len(daat_skip_tfidf_results)
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_comparisons'] = daat_skip_tfidf_comparisons


        return output_dict


@app.route("/execute_query", methods=['POST'])
def execute_query():
    """ This function handles the POST request to your endpoint.
        Do NOT change it."""
    start_time = time.time()

    queries = request.json["queries"]
    # random_command = request.json["random_command"] #Random command removed as per piazza post

    """ Running the queries against the pre-loaded index. """
    output_dict = runner.run_queries(queries)

    """ Dumping the results to a JSON file. """
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.Response(
        json.dumps(response, indent=4),
        mimetype='application/json'
    )


if __name__ == "__main__":
    """ Driver code for the project, which defines the global variables.
        Do NOT change it."""

    output_location = "project2_output.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=str, default = "data/input_corpus.txt", help="Corpus File name, with path.")
    parser.add_argument("--output_location", type=str, help="Output file name.", default=output_location)
    parser.add_argument("--username", default = "dsingh27", type=str,
                        help="Your UB username. It's the part of your UB email id before the @buffalo.edu. "
                             "DO NOT pass incorrect value here")

    argv = parser.parse_args()

    corpus = argv.corpus
    output_location = argv.output_location
    username_hash = hashlib.md5(argv.username.encode()).hexdigest()

    """ Initialize the project runner"""
    runner = ProjectRunner()

    """ Index the documents from beforehand. When the API endpoint is hit, queries are run against 
        this pre-loaded in memory index. """
    runner.run_indexer(corpus)

    app.run(host="0.0.0.0", port=9999)
