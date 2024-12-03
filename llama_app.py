# app2.py
# export FLASK_ENV=production - write this before running the code
import os
import json
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

#from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

import re
import string
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")  # Ignore all the warnings, avoid clutter in console output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")


# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')  # Replace with a secure secret key

# Configure server-side session (optional but recommended for scalability)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Paths to data files
INVERTED_INDEX_PATH = 'inverted_index.json'
METADATA_PATH = 'metadata.json'
SCRAPPED_DATA_PATH = 'final_scrapped.json'


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the Llama model and tokenizer
class LlamaModel:
    def __init__(self, model_name="meta-llama/Llama-3.2-1b-instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).half().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(self.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=500,
            temperature=0.2,  # Lowered for more deterministic responses
            top_p=0.9,
            top_k=50,
            do_sample=True,  # Enable sampling
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


# Initialize the Llama model
llama_model = LlamaModel()

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def tokenize(self, text):
        sentence = text.lower()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()  
        words = sentence.split()
        words = [w for w in words if not w in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in words]
        return tokens
    
class DocumentRetriever:
    def __init__(self, inverted_index_path, metadata_path, scrapped_data_path):
        self.preprocessor = Preprocessor()
        self.inverted_index = self.load_inverted_index(inverted_index_path)
        self.metadata = self.load_metadata(metadata_path)
        self.doc_id_to_summary = self.load_scrapped_data(scrapped_data_path)

    def load_inverted_index(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                inverted_index = json.load(f)
            #print(f"Inverted index loaded from '{path}'.")
            return inverted_index
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def load_metadata(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            #print(f"Metadata loaded from '{path}'.")
            return metadata
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def load_scrapped_data(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            doc_id_to_summary = {}
            for item in data:
                revision_id = item.get('revision_id')
                summary = item.get('summary', '')
                url = item.get('url', 'URL not found')  # Include the URL
                if revision_id:
                    doc_id_to_summary[revision_id] = {
                        'summary':summary,
                        'url': url
                    }
            #print(f"Scrapped data loaded from '{path}'.")
            return doc_id_to_summary
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def _daat_and_with_tfidf(self, terms):
        inverted_index = self.inverted_index
        postings_lists = []
        for term in terms:
            if term in inverted_index:
                postings = inverted_index[term]
                postings_lists.append(set([posting['doc_id'] for posting in postings]))
            else:
                # If any term is not in the index, intersection is empty
                return [], 0
            
        intersection = set.intersection(*postings_lists) if postings_lists else set()
        total_comparisons = sum(len(p) for p in postings_lists)

        if not intersection:
            return [], total_comparisons

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

    def retrieve_top_k(self, query, k=3):
        tokens = self.preprocessor.tokenize(query)
        if not tokens:
            print("No valid terms found in the query after preprocessing.")
            return []
        
        print(f"Processed Query Terms: {tokens}")
        ranked_docs, comparisons = self._daat_and_with_tfidf(tokens)
        top_k_docs = ranked_docs[:k]
        summaries = []
        for doc_id, score in top_k_docs:
            summary = self.doc_id_to_summary.get(doc_id, "No summary available.")
            summaries.append({
                'doc_id': doc_id,
                'summary': summary,
                'score': score
            })

        return summaries, comparisons, tokens



# Initialize the DocumentRetriever
retriever = DocumentRetriever(INVERTED_INDEX_PATH, METADATA_PATH, SCRAPPED_DATA_PATH)

# Paraphrase question
def rephrase_question_with_history(chat_history, question):
    prompt = f"""
    <BEGIN CONVERSATION>
    <SYSTEM>
    You are an assistant that uses chat history to rephrase the question to generate 5 standalone queries, that  rephrases follow-up questions to be standalone questions, containing only important query terms.
    </SYSTEM>
    
    <CONVERSATION HISTORY>
    {chat_history}
    </CONVERSATION HISTORY>
    
    <FOLLOW UP INPUT>
    {question}
    </FOLLOW UP INPUT>
    
    <STANDALONE QUESTIONS>
    """
    response = llama_model.generate_response(prompt)
    if response:
        return response.strip()
    else:
        return "Error: Response content not found."


# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="conversation_history")

# Initialize Chains
def determine_intent(conversation_history, user_input):
    prompt = f"""
    <BEGIN CONVERSATION>
    <SYSTEM>
    You are an intelligent assistant. Determine the intent of the user's input based on the conversation history. Answer with either "chitchat" or "query".
    If the user's input contains gratitude expressions like "thanks," "thank you," "thanks a lot," or "much appreciated," classify it as "chitchat."
    </SYSTEM>
    
    <CONVERSATION HISTORY>
    {conversation_history}
    </CONVERSATION HISTORY>
    
    <USER INPUT>
    {user_input}
    </USER INPUT>
    
    <INTENT>
    """
    return llama_model.generate_response(prompt).strip()


def gen_response(context, user_input):
    prompt = f"""
    <BEGIN CONVERSATION>
    <SYSTEM>
    You are an expert researcher with knowledge based on Wikipedia. Use the following context to answer the user's question accurately and concisely. Handle edge cases effectively. If there's only one relevant result, formulate the answer similarly to the question asked. If no relevant data is found, inform the user without providing an answer. Always provide URLs at the end to indicate references used.
    </SYSTEM>
    
    <CONTEXT>
    {context}
    </CONTEXT>
    
    <USER INPUT>
    {user_input}
    </USER INPUT>
    
    <ANSWER>
    """
    return llama_model.generate_response(prompt)



def handle_chitchat(conversation_history, user_input):
    prompt = f"""
    <BEGIN CONVERSATION>
    <SYSTEM>
    You are a friendly and intelligent assistant. Engage in a natural and coherent conversation based on the user's input.
    </SYSTEM>
    
    <CONVERSATION HISTORY>
    {conversation_history}
    </CONVERSATION HISTORY>
    
    <USER INPUT>
    {user_input}
    </USER INPUT>
    
    <ASSISTANT RESPONSE>
    """
    return llama_model.generate_response(prompt)




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': "I didn't receive any input."})

    memory.save_context({"user_input": user_input}, {"user_input": user_input})
    conversation_history = memory.load_memory_variables({})['conversation_history']
    #print(f'conversation history is {conversation_history}')

    # Step 1: Determine Intent
    intent = determine_intent(conversation_history, user_input).lower()
    print(f"Determined Intent: {intent}")
    tfidf_data = []
    if 'query' in intent:
        # Informational Query Handling
        standalone_question = rephrase_question_with_history(conversation_history, user_input)
        print(f'paraphrased question is: {standalone_question}')
        questions = standalone_question.split('|')
        total_context = ""
        total_q = ""
        for q in questions:
            summaries, comparisons, tokens = retriever.retrieve_top_k(q, k=5)
            if summaries:
                context = ""
                for idx, doc in enumerate(summaries, start=1):
                    #url = retriever.doc_id_to_summary.get(doc['doc_id'], {}).get('url', 'URL not found')
                    context += f"Document{idx} (Doc ID: {doc['doc_id']}): {doc['summary']}\n"
                tfidf_data.append({
                    'query':q,
                    'tfidf_scores':summaries,
                    'comparisons':comparisons,
                    'tokens':tokens
                })
            else:
                context = "No relevant information found for this question\n"
            total_context += f"rephrased question:{q} \n context:{context}"
            total_q += q
        print("Total context:\n", total_context)
        print("Total Q:\n", total_q)
        print("Actual question:\n", questions[0])
        answer = generate_response(total_context, total_q)
    else:
        # Chitchat Handling
        answer = handle_chitchat(conversation_history, user_input)

    # Append assistant response to memory
    memory.save_context({"assistant_output": answer}, {"assistant_output": answer})

    return jsonify({
        'response': answer,
        'tfidf_data':tfidf_data
        })

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000, debug=True)



