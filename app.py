from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask app
app = Flask(__name__)

# Load .env (if needed for Pinecone)
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download HF embedding model
embeddings = download_hugging_face_embeddings()

# Connect to existing Pinecone index
index_name = "testbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever from Pinecone
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Initialize Ollama LLM
llm = OllamaLLM(model="llama3")

# Prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# RAG chain setup
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    try:
        print("Before RAG invoke")
        response = rag_chain.invoke({"input": msg})
        print("After RAG invoke")
        print("Response:", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print("Error:", e)
        return "Sorry, there was an error processing your request."

# Start Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
