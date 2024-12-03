# app2.py
# export FLASK_ENV=production - write this before running the code
import os
import json
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
#from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from transformers import pipeline

import re
import string
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")  # Ignore all the warnings, avoid clutter in console output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")


# Load environment variables
load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')

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
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = [
        "Environment", "Health", "Economy", "Technology", 
        "Entertainment", "Sports", "Education", "Politics and Government", 
        "Food", "Travel"
    ]

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

    def _daat_and_with_tfidf(self, terms, labels): # uses labels to filter
        inverted_index = self.inverted_index
        # Initialize postings lists for each term
        postings_lists = []
        for term in terms:
            if term in inverted_index:
                postings = inverted_index[term]
                postings_lists.append(set([posting['doc_id'] for posting in postings if posting['topic'].lower() in labels]))
                #postings_lists.append(set([posting['doc_id'] for posting in postings]))
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

    def retrieve_top_k(self, query, labels, k=3):
        # Preprocess the query
        tokens = self.preprocessor.tokenize(query)
        if not tokens:
            print("No valid terms found in the query after preprocessing.")
            return []
        print(f"Processed Query Terms: {tokens}")
        ranked_docs, comparisons = self._daat_and_with_tfidf(tokens,labels)

        # Get top-k documents
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

# Initialize Langchain's Ollama LLM
llm = ChatOpenAI(model=OLLAMA_MODEL, temperature=0.2)  # Adjust temperature as needed

def classify_query_top_k(query,select_topics, k):
        candidate_labels = [
            "Environment", "Health", "Economy", "Technology", 
            "Entertainment", "Sports", "Education", "Politics and Government", 
            "Food", "Travel"
        ]
        if len(select_topics) > 0:
            candidate_labels = select_topics
        result = classifier(query, candidate_labels, multi_label=True, hypothesis_template="This text is about {}.")
        top_predictions = sorted(result["labels"], key=lambda x: result["scores"][result["labels"].index(x)], reverse=True)[:k]
        return top_predictions

# Paraphrase question
def rephrase_question_with_history(chat_history, question):
    # currently takes only the last question as chat history. Need to change this to append all previous convos for this session as chat.
    template = """
    Rephrase the following question to 5 standalone queries, each separated by '|', suitable for information retrieval. Ensure it contains all necessary context without relying on previous conversations.
    If the user's input contains gratitude expressions like "thanks," "thank you," "thanks a lot," or "much appreciated,", the generated query is the same as the question.
    <chat_history>
      {chat_history}
    </chat_history>
    
    Follow Up Input: {question}
    Standalone questions:
    """
    system_message = "You are an assistant that rephrases follow-up questions to be standalone questions, containing only important query terms."
    llm = ChatOpenAI(model=OLLAMA_MODEL, temperature=0.2)
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(chat_history=chat_history, question=question)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_prompt}
    ]
    # why not just pass the messages dict as the chat history?
    response = llm.invoke(messages)
    content = response.content
    if content:
            return content.strip()  
    else:
            return "Error: Response content not found."

# Intent Classification Prompt
intent_template = """
You are an intelligent assistant. Determine the intent of the user's input based on the conversation history. Answer with either "chitchat" or "query".
If the user's input contains gratitude expressions like "thanks," "thank you," "thanks a lot," or "much appreciated," classify it as "chitchat."

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
You are an expert researcher with knowledge based on wikipedia. Use the following context to answer the user's question accurately and concisely. You are smart enough to handle edge cases. Like if there is only one relevant result, formulate it to generate an answer similar to the question asked. If there is no relavant data found, inform the user, and don't return an answer, else answer the question in coherence to the first question asked. Provide urls at the end, to indicate references used.

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
    selected_topics = request.json.get('selected_topics', []) 
    if not user_input:
        return jsonify({'response': "I didn't receive any input."})

    memory.save_context({"user_input": user_input}, {"user_input": user_input})
    conversation_history = memory.load_memory_variables({})['conversation_history']
    #print(f'conversation history is {conversation_history}')

    # Step 1: Determine Intent
    intent = intent_chain.invoke({
        "conversation_history": conversation_history,
        "user_input": user_input
    })["text"]
    intent = intent.strip().lower()
    print(f"Determined Intent: {intent}")
    tfidf_data = []
    labels_data = []
    if 'query' in intent:
        # Informational Query Handling
        standalone_question = rephrase_question_with_history(conversation_history, user_input)
        print(f'paraphrased question is: {standalone_question}')
        questions = standalone_question.split('|')
        total_context = ""
        temp = [q.strip() for q in questions]
        total_q = " ".join(temp)
        labels = classify_query_top_k(total_q,selected_topics,k=3)
        labels = [label.lower() for label in labels]
        print("Labels:",labels)
        labels_data = labels
        for q in questions:
            summaries, comparisons, tokens = retriever.retrieve_top_k(q, labels, k=5)
            if summaries:
                context = ""
                for idx, doc in enumerate(summaries, start=1):
                    url = retriever.doc_id_to_summary.get(doc['doc_id'], {}).get('url', 'URL not found')
                    context += f"Document{idx} (Doc ID: {doc['doc_id']}): {doc['summary']}\n"
                tfidf_data.append({
                    'query':q,
                    'tfidf_scores':summaries,
                    'comparisons':comparisons,
                    'tokens':tokens 
                })
                #labels_data.append(labels)
            else:
                context = "No relevant information found for this question\n"
            total_context += f"rephrased question:{q} \n context:{context}"
            total_q += q
        print("Total context:\n", total_context)
        print("Total Q:\n", total_q)
        print("Actual question:\n", questions[0])
        breakpoint
        answer = response_chain.invoke({
            "context": total_context,
            "user_input": total_q
        })["text"]
    else:
        # Chitchat Handling
        answer = chitchat_chain.invoke({
            "conversation_history": conversation_history,
            "user_input": user_input
        })["text"]

    # Append assistant response to memory
    memory.save_context({"assistant_output": answer}, {"assistant_output": answer})
    print("Sending the labels: ",labels_data)
    return jsonify({
        'response': answer,
        'tfidf_data':tfidf_data,
        'labels_data':labels_data
        })

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000, debug=True)



