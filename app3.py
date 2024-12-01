# app2.py
# To run: export FLASK_ENV=production && python app2.py

import os
import json
import re
import string
import logging
from collections import defaultdict
from functools import lru_cache

from flask import Flask, render_template, request, jsonify
from flask_session import Session
from markupsafe import escape

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Suppress specific warnings to keep console output clean
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")

# ================================
# Configuration and Setup
# ================================

# Initialize NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application")

# Configure server-side session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Paths to data files (can be set via environment variables)
INVERTED_INDEX_PATH = os.getenv('INVERTED_INDEX_PATH', 'inverted_index.json')
METADATA_PATH = os.getenv('METADATA_PATH', 'metadata.json')
SCRAPPED_DATA_PATH = os.getenv('SCRAPPED_DATA_PATH', 'final_scrapped.json')

# Configuration for GPT-4o-mini via Ollama
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gpt-4o-mini')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY set for Flask application")

# ================================
# Preprocessor Class
# ================================

class Preprocessor:
    """
    Preprocesses text by tokenizing, removing stopwords, and lemmatizing.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def tokenize(self, text):
        # Convert to lowercase
        sentence = text.lower()
        # Remove non-alphanumeric characters
        sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
        # Remove extra spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        # Split into words
        words = sentence.split()
        # Remove stopwords
        words = [w for w in words if w not in self.stop_words]
        # Lemmatize words
        tokens = [self.lemmatizer.lemmatize(token) for token in words]
        return tokens

# ================================
# DocumentRetriever Class
# ================================

class DocumentRetriever:
    """
    Retrieves and ranks documents based on TF-IDF scores.
    """
    def __init__(self, inverted_index_path, metadata_path, scrapped_data_path):
        self.preprocessor = Preprocessor()
        self.inverted_index = self.load_json(inverted_index_path, "Inverted index")
        self.metadata = self.load_json(metadata_path, "Metadata")
        self.doc_id_to_info = self.load_scrapped_data(scrapped_data_path)
    
    def load_json(self, path, description):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"{description} loaded from '{path}'.")
            return data
        except FileNotFoundError:
            logger.error(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def load_scrapped_data(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            doc_id_to_info = {}
            for item in data:
                revision_id = item.get('revision_id')
                summary = item.get('summary', '')
                url = item.get('url', '')
                if revision_id:
                    doc_id_to_info[revision_id] = {'summary': summary, 'url': url}
            logger.info(f"Scrapped data loaded from '{path}'.")
            return doc_id_to_info
        except FileNotFoundError:
            logger.error(f"Error: The file '{path}' was not found.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error: The file '{path}' does not contain valid JSON.")
            return {}

    def _daat_and_with_tfidf(self, terms):
        inverted_index = self.inverted_index
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

    @lru_cache(maxsize=1000)
    def retrieve_top_k(self, query, k=5):
        """
        Retrieves top-k relevant documents for the given query.
        Utilizes caching to optimize performance for frequent queries.
        """
        tokens = self.preprocessor.tokenize(query)
        if not tokens:
            logger.info("No valid terms found in the query after preprocessing.")
            return []

        logger.info(f"Processed Query Terms: {tokens}")
        ranked_docs, comparisons = self._daat_and_with_tfidf(tuple(tokens))
        top_k_docs = ranked_docs[:k]

        summaries = []
        for doc_id, score in top_k_docs:
            info = self.doc_id_to_info.get(doc_id, {})
            summary = info.get('summary', "No summary available.")
            url = info.get('url', "No URL available.")
            summaries.append({
                'doc_id': doc_id,
                'summary': summary,
                'url': url,
                'score': score
            })

        logger.info(f"Retrieved {len(summaries)} documents with total comparisons: {comparisons}")
        return summaries

# ================================
# Initialize DocumentRetriever
# ================================

retriever = DocumentRetriever(INVERTED_INDEX_PATH, METADATA_PATH, SCRAPPED_DATA_PATH)

# ================================
# Initialize Langchain's LLM
# ================================

llm = ChatOpenAI(model=OLLAMA_MODEL, temperature=0.2)  # Adjust temperature as needed

# ================================
# Rephrase Question Function
# ================================

def rephrase_question_with_history(chat_history, question):
    """
    Rephrases a follow-up question into a standalone question by incorporating necessary context from the conversation history.
    """
    template = """
    Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone question that includes all necessary context and relevant terms from the conversation history.

    <Conversation History>
    {chat_history}
    </Conversation History>

    <Follow-Up Question>
    {question}
    </Follow-Up Question>

    Rephrased Standalone Question:
    """
    system_message = "You are an assistant that rephrases follow-up questions to be standalone questions, incorporating necessary context from the conversation history."

    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(chat_history=chat_history, question=question)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_prompt}
    ]

    response = llm.invoke(messages)
    content = response.content
    if content:
        # Assuming the LLM returns a single standalone question
        return content.strip()
    else:
        return "Error: Unable to rephrase the question."

# ================================
# Define Prompt Templates
# ================================

# Intent Classification Prompt
intent_template = """
You are an intelligent assistant. Determine the intent of the user's input based on the conversation history. Answer with either "chitchat" or "query".

Conversation History:
{conversation_history}

User: {user_input}
Intent (chitchat/query):
"""

intent_prompt = PromptTemplate(
    input_variables=["conversation_history", "user_input"],
    template=intent_template,
)

# Response Generation Prompt for Informational Queries with References
response_template = """
You are an expert researcher with knowledge based on the provided document summaries. Using the summaries below, generate an accurate and concise answer to the user's question. After the answer, list the references used in the format 'References: Doc1 (url1), Doc2 (url2), Doc3 (url3)'.

### Question:
{user_input}

### Document Summaries:
{context}

### Answer:
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

# ================================
# Initialize Conversation Memory
# ================================

memory = ConversationBufferMemory(memory_key="conversation_history")

# ================================
# Initialize LLM Chains
# ================================

intent_chain = LLMChain(
    llm=llm,
    prompt=intent_prompt,
    verbose=False,
    memory=memory
)

response_chain = LLMChain(
    llm=llm,
    prompt=response_prompt,
    verbose=False  # Disable verbose logging for production
)

chitchat_chain = LLMChain(
    llm=llm,
    prompt=chitchat_prompt,
    verbose=False,
    memory=memory
)

# ================================
# Flask Routes
# ================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        # Sanitize user input to prevent injection attacks
        user_input = escape(request.json.get('message', ''))
        if not user_input:
            return jsonify({'response': "I didn't receive any input."})

        # Append user input to memory
        memory.save_context({"user_input": user_input}, {"user_input": user_input})

        # Retrieve conversation history
        conversation_history = memory.load_memory_variables({})['conversation_history']

        # Rephrase question to include necessary context
        standalone_question = rephrase_question_with_history(conversation_history, user_input)

        logger.info(f'Paraphrased Standalone Question: {standalone_question}')

        # Determine Intent
        intent = intent_chain.invoke({
            "conversation_history": conversation_history,
            "user_input": standalone_question
        })["text"].strip().lower()
        logger.info(f"Determined Intent: {intent}")

        if 'query' in intent:
            # Informational Query Handling
            summaries = retriever.retrieve_top_k(standalone_question, k=5)
            if summaries:
                # Aggregate summaries and collect reference IDs with URLs
                summaries_text = "\n".join([
                    f"Doc {i+1} (Doc ID: {doc['doc_id']}): {doc['summary']}" 
                    for i, doc in enumerate(summaries)
                ])
                references = ", ".join([
                    f"{doc['doc_id']} ({doc['url']})" for doc in summaries
                ])

                # Generate response using LLM
                answer = response_chain.invoke({
                    "context": summaries_text,
                    "user_input": standalone_question
                })["text"]

                # Append References
                answer += f"\n\nReferences: {references}"
            else:
                # Handle no relevant information
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

    except Exception as e:
        logger.exception("Error processing the request.")
        return jsonify({'response': "An error occurred while processing your request."}), 500

# ================================
# Error Handlers
# ================================

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}")
    return jsonify({'response': "An unexpected error occurred. Please try again later."}), 500

@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"Not Found: {error}")
    return jsonify({'response': "The requested resource was not found."}), 404

# ================================
# Helper Functions
# ================================

def save_to_memory(user_input, assistant_output, max_history=10):
    """
    Saves the latest user and assistant interactions to the conversation memory,
    ensuring that the history does not exceed the maximum allowed length.
    """
    history = conversation_history_split(conversation_history=memory.load_memory_variables({})['conversation_history'])
    if len(history) > max_history * 2:  # Each interaction has user and assistant
        history = history[-max_history * 2:]
    history.append(f"User: {user_input}")
    history.append(f"Assistant: {assistant_output}")
    updated_history = "\n".join(history)
    memory.save_context({"conversation_history": updated_history}, {"conversation_history": updated_history})

def conversation_history_split(conversation_history):
    """
    Splits the conversation history into individual lines.
    """
    return conversation_history.split('\n') if conversation_history else []

# ================================
# Main Execution
# ================================

if __name__ == '__main__':
    app.run(debug=True)
