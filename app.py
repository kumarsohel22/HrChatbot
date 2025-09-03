from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from dotenv import load_dotenv

load_dotenv()

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
ollama_llm = OllamaLLM(model="llama2")

# Setup QA chain
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
<context>
{context}
</context>
Question: {input}
""")
document_chain = create_stuff_documents_chain(llm=ollama_llm, prompt=prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# FastAPI app
app = FastAPI(title="HR Chat API")

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
async def search_employees(skill: str = Query(..., min_length=1, description="Skill to search for")):
    try:
        matched = []
        for doc in docs:
            if skill.lower() in str(doc.page_content).lower():
                matched.append(str(doc.page_content))
        return {"results": matched[:10]}  # return top 10 matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
