# app2.py

import os
import json
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

import re
import string
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load environment variables
load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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

# Configuration for GPT-4o-mini via Ollama
OLLAMA_MODEL = 'gpt-4o-mini'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # self.stemmer = PorterStemmer()
        self.ps = PorterStemmer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def tokenize(self, text):
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
        Returns a list of tuples (doc_id, cumulative_tfidf) sorted by tfidf descending.
        """
        inverted_index = self.inverted_index
        # metadata = self.metadata
        # total_documents = metadata.get('total_documents', 0)

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

    def retrieve_top_k(self, query, k=3):
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
        # print(f"Top {k} Documents:")
        # for rank, (doc_id, score) in enumerate(top_k_docs, start=1):
            # print(f"{rank}. Doc ID: {doc_id}, TF-IDF Score: {score:.4f}")

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



# Initialize the DocumentRetriever
retriever = DocumentRetriever(INVERTED_INDEX_PATH, METADATA_PATH, SCRAPPED_DATA_PATH)

# Initialize Langchain's Ollama LLM
llm = ChatOpenAI(model=OLLAMA_MODEL, temperature=0.2)  # Adjust temperature as needed

# Paraphrase question
def rephrase_question_with_history(chat_history, question):
    template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    
    <chat_history>
      {chat_history}
    </chat_history>
    
    Follow Up Input: {question}
    Standalone question:
    """

    # Define system message if necessary
    system_message = "You are an assistant that rephrases follow-up questions to be standalone questions."

    # Initialize the ChatOpenAI LLM
    llm = ChatOpenAI(model=OLLAMA_MODEL, temperature=0.2)

    # Create PromptTemplate
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(chat_history=chat_history, question=question)

    # Prepare messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_prompt}
    ]

    # Invoke the LLM
    response = llm.invoke(messages)

    print(f'response is {response}')
    return response['choices'][0]['message']['content'] if 'choices' in response else str(response)

# Define Prompt Templates

# Intent Classification Prompt
intent_template = """
You are an intelligent assistant. Determine the intent of the user's input. Answer with either "chitchat" or "query".

Conversation History:
{conversation_history}

User: {user_input}
Intent (chitchat/query):
"""

intent_prompt = PromptTemplate(
    input_variables=["conversation_history", "user_input"],
    template=intent_template,
)

# Response Generation Prompt for Informational Queries
response_template = """
You are an expert researcher. Use the following context to answer the user's question accurately and concisely.

Context:
{context}

User: {user_input}
Answer:
"""

response_prompt = PromptTemplate(
    input_variables=["context", "user_input"],
    template=response_template,
)

# Chitchat Prompt
chitchat_template = """
You are a friendly and intelligent assistant. Engage in a natural and coherent conversation based on the user's input.

Conversation History:
{conversation_history}

User: {user_input}
Assistant:
"""

chitchat_prompt = PromptTemplate(
    input_variables=["conversation_history", "user_input"],
    template=chitchat_template,
)

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="conversation_history")

# Initialize Chains
intent_chain = LLMChain(
    llm=llm,
    prompt=intent_prompt,
    verbose=False,
    memory=memory
)

response_chain = LLMChain(
    llm=llm,
    prompt=response_prompt,
    verbose=False  # Removed memory=memory
)

chitchat_chain = LLMChain(
    llm=llm,
    prompt=chitchat_prompt,
    verbose=False,
    memory=memory
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': "I didn't receive any input."})

    # Append user input to memory
    memory.save_context({"user_input": user_input}, {"user_input": user_input})

    # Retrieve conversation history
    conversation_history = memory.load_memory_variables({})['conversation_history']
    print(f'conversation history is {conversation_history}')
    standalone_question = rephrase_question_with_history(conversation_history, user_input)

    print(f'paraphrased question is: {standalone_question}')
    # Step 1: Determine Intent
    intent = intent_chain.invoke({
        "conversation_history": conversation_history,
        "user_input": standalone_question
    })["text"]
    intent = intent.strip().lower()
    print(f"Determined Intent: {intent}")

    if 'query' in intent:
        # Informational Query Handling
        summaries = retriever.retrieve_top_k(user_input, k=10)

        if summaries:
            # Prepare context from summaries
            context = ""
            for idx, doc in enumerate(summaries, start=1):
                context += f"Document {idx} (Doc ID: {doc['doc_id']}): {doc['summary']}\n"

            # Generate response using context
            answer = response_chain.invoke({
                "context": context,
                "user_input": standalone_question
            })["text"]
        else:
            answer = "I'm sorry, I couldn't find any relevant information on that topic."
    else:
        # Chitchat Handling
        answer = chitchat_chain.invoke({
            "conversation_history": conversation_history,
            "user_input": standalone_question
        })["text"]

    # Append assistant response to memory
    memory.save_context({"assistant_output": answer}, {"assistant_output": answer})

    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)
