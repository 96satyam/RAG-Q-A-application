import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langserve import add_routes
import uvicorn
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time



load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="RAG Q&A Server",
    version="1.0",
    description="RAG Q&A Chatbot Server"
)

class QueryModel(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    report_type: str

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("report")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

document_chain = None
retriever = None

def setup_retrieval_chain():
    global document_chain, retriever
    if document_chain is None or retriever is None:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()

@app.post("/ask")
def ask_question(query: QueryModel):
    if "vectors" not in st.session_state:
        raise HTTPException(status_code=400, detail="Vector database is not initialized. Run embedding first.")
    
    setup_retrieval_chain()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': query.question})
    return {"answer": response['answer']}

@app.post("/analytics")
def generate_analytics(request: AnalyticsRequest):
    # Placeholder for analytics logic
    return {"message": f"Analytics report generated for {request.report_type}"}

add_routes(app, ChatGroq(), path="/groqai")
add_routes(app, ChatPromptTemplate.from_template("{input}") | ChatGroq(), path="/ask")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


