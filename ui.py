# ui.py
# imports   
import streamlit as st
import requests
import threading
import time
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ----------------- FastAPI backend -----------------
app = FastAPI(title="HR Chat API")

# Load documents & build vectorstore
loader = UnstructuredExcelLoader(r"D:\STUDY\Gen AI\Langchain\hrchatbot\employees.xlsx", mode="elements")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Load LLMs
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

chat_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
ollama_llm = Ollama(model="llama2")

# Setup QA chain
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
<context>
{context}
</context>
Question: {input}
""")
document_chain = create_stuff_documents_chain(llm=chat_llm, prompt=prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Pydantic models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = retrieval_chain.invoke({"input": request.query})
        return {"answer": result.get("answer", "No answer found")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employees/search")
async def search_employees(skill: str = Query(..., min_length=1)):
    try:
        matched = []
        for doc in docs:
            if skill.lower() in str(doc.page_content).lower():
                matched.append(str(doc.page_content))
        return {"results": matched[:10]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Run FastAPI in background -----------------
def run_backend():
    uvicorn.run(app, host="127.0.0.1", port=8000)

threading.Thread(target=run_backend, daemon=True).start()
time.sleep(2)  # Give backend time to start

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="HR Chatbot", layout="wide")
st.title("💼 HR Chatbot")

st.sidebar.header("Search Employees")
skill = st.sidebar.text_input("Skill to search for")
if st.sidebar.button("Search"):
    if skill:
        try:
            response = requests.get("http://127.0.0.1:8000/employees/search", params={"skill": skill})
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    for emp in results:
                        st.write(emp)
                else:
                    st.info("No employees found with this skill.")
            else:
                st.error("Error fetching employees")
        except requests.exceptions.RequestException as e:
            st.error(f"Backend not reachable: {e}")

st.header("Chat with HR Assistant")
query = st.text_input("Ask a question about employees/projects/etc.")
if st.button("Send"):
    if query:
        try:
            response = requests.post("http://127.0.0.1:8000/chat", json={"query": query})
            if response.status_code == 200:
                answer = response.json().get("answer", "")
                st.success(answer)
            else:
                st.error("Error processing your query")
        except requests.exceptions.RequestException as e:
            st.error(f"Backend not reachable: {e}")
