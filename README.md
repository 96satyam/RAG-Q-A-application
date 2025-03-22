# RAG-Q-A-application

## RAG Q&A Chatbot Server

### Overview

This project implements a Retrieval-Augmented Generation (RAG) Q&A Chatbot Server using FastAPI, Streamlit, and LangChain. The chatbot leverages Groq's LLaMA 3.3-70B model for answering user queries based on a provided knowledge base. The system also integrates FAISS vector embeddings for document retrieval and OpenAI embeddings for efficient text processing.

### Features

#### RAG-based chatbot: Answers user queries based on retrieved documents.

#### FastAPI-powered backend: Provides a RESTful API for handling chatbot queries and analytics requests.

#### Streamlit integration: Manages session-based embeddings and vector storage.

#### FAISS vector database: Efficient document retrieval for accurate responses.

#### OpenAI & Groq integration: Utilizes OpenAI embeddings and Groq’s LLaMA model for natural language processing.

#### PDF document support: Loads documents from a directory for knowledge extraction.

### Tech Stack

#### Python (FastAPI, Streamlit, LangChain, FAISS, OpenAI, Groq)

#### FastAPI for building the RESTful backend

#### Streamlit for session management and embedding initialization

#### LangChain for retrieval and response generation

#### FAISS for vector-based search and retrieval

#### Uvicorn for running the API server

#### Installation & Setup

#### Prerequisites

#### Python 3.8+

#### OpenAI API Key

#### Groq API Key

#### Required dependencies listed in requirements.txt

#### Steps to Run the Project

#### Clone the Repository

#### git clone https://github.com/96satyam/RAG-Q-A-application
cd yourrepo

#### Create a Virtual Environment

### python -m venv venv
##### source venv/bin/activate   # On Windows use: venv\Scripts\activate

### Install Dependencies

##### pip install -r requirements.txt

#### Set Environment Variables

#### Create a .env file and add your API keys:

#### OPENAI_API_KEY=your_openai_api_key
#### GROQ_API_KEY=your_groq_api_key

#### Run the FastAPI Server

#### uvicorn app:app --host localhost --port 8000

#### Run the Streamlit App (Optional)

##### streamlit run app.py

#### API Endpoints

#### 1. Ask a Question

#### Endpoint: /ask

##### Method: POST

Request Body:

{
  "question": "What is AI?"
}

#### Response:

{
  "answer": "AI stands for Artificial Intelligence, which refers to..."
}

#### 2. Generate Analytics Report

Endpoint: /analytics

Method: POST

Request Body:

{
  "report_type": "user_engagement"
}

Response:

{
  "message": "Analytics report generated for user_engagement"
}

### File Structure

##### ├── app.py               # Main application file
##### ├── analysisdata.ipynb   # Data analysis notebook
##### ├── datapreprocessing.ipynb # Data preprocessing steps
##### ├── requirements.txt     # Python dependencies
##### ├── .env                 # API keys (not included in repo)
##### └── README.md            # Project documentation

#### Future Improvements

Implement UI for user-friendly chatbot interaction.

Add more advanced analytics features.

Extend document support beyond PDFs.

Deploy the system on a cloud platform.
