# Trinity_AI_internship
# Metro City Smart Information Assistant

This project implements a Smart City Information Assistant using a Retrieval Augmented Generation (RAG) pipeline, a FastAPI backend, a Streamlit frontend, and an optional multi-agent system built with CrewAI. It helps citizens get instant answers about city services, public facilities, transportation, policies, and emergency procedures.

## Table of Contents

1.  [Features](#features)
2.  [Project Structure](#project-structure)
5.  [Setup Instructions](#setup-instructions)
    *   [1. Clone the Repository](#1-clone-the-repository)
    *   [2. Create a Virtual Environment](#2-create-a-virtual-environment)
    *   [3. Install Dependencies](#3-install-dependencies)
6.  [Running the Application](#running-the-application)
    *   [1. Start the FastAPI Backend](#1-start-the-fastapi-backend)
    *   [2. Start the Streamlit Frontend](#2-start-the-streamlit-frontend)
7.  [Usage](#usage)
    *   [Streamlit Frontend](#streamlit-frontend)
    *   [API Endpoints](#api-endpoints)
8.  [Knowledge Base Usage (`knowledge.json`)](#knowledge-base-usage-knowledgejson)
    *   [Data Loading and Processing](#data-loading-and-processing)
    *   [Chunking](#chunking)
    *   [Embedding and Indexing](#embedding-and-indexing)
9.  [API Documentation (FastAPI)](#api-documentation-fastapi)
    *   [`GET /health`](#get-health)
    *   [`POST /search`](#post-search)
    *   [`POST /query`](#post-query)
    *   [`POST /crew-ask`](#post-crew-ask)


## Features

*   **RAG Pipeline:** Utilizes a Retrieval Augmented Generation pipeline to provide answers grounded in a local knowledge base.
*   **FastAPI Backend:** Exposes robust API endpoints for querying, searching, and health checks.
*   **Streamlit Frontend:** Offers a user-friendly interface to interact with the assistant.
*   **Vector Search:** Allows direct semantic search over the knowledge base.
*   **CrewAI Integration (Optional):** Demonstrates a multi-agent approach for handling complex queries with specialized agents.
*   **Local Embeddings:** Uses Hugging Face sentence transformers for local text embedding.

## Project Structure
```markdown
smart_city_assistant/
├── app/ # FastAPI application and RAG logic
│ ├── main.py # FastAPI app, endpoints, RAG initialization
│ ├── models.py # Pydantic models for API requests/responses
│ └── crew_tools.py # Tools for CrewAI agents (if using CrewAI)
├── frontend.py # Streamlit frontend application
├── knowledge.json # The core knowledge base for the assistant
├── faiss_index_city_info_hf/ # Stores the pre-computed FAISS vector index
│ └── index.faiss
│ └── index.pkl
├── README.md # This file
└── requirements.txt # Python package dependencies 
```
## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd smart_city_assistant
```
### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
Install using pip:
```bash
pip install requirements.txt
```
## Running the Application
You need to run the backend and frontend separately, typically in two different terminal windows.
### 1. Start the FastAPI Backend
Navigate to the project's root directory (smart_city_assistant/) where the app/ folder is located.
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
### 2. Start the Streamlit Frontend
In a new terminal, navigate to the project's root directory.
```bash
streamlit run frontend.py
```
## Usage
### Streamlit Frontend
Once both backend and frontend are running:
*  Open your web browser to the Streamlit URL (e.g., http://localhost:8501).
*  You'll see the "Metro City Information Assistant" interface.
*  Use the sidebar to choose a mode:
*  Chat with Assistant (/query): Type your question and click "Ask Assistant" to get an LLM-generated answer based on retrieved knowledge.
*  Direct Vector Search (/search): Type your query, set k (number of documents to retrieve), and click "Search Documents" to see raw relevant chunks from the knowledge base.
*  Ask the Crew (/crew-ask) (If enabled): Type a task description and submit it to the CrewAI agents.
*  View the results displayed on the page.

## Knowledge Base Usage (knowledge.json)
The knowledge.json file is the heart of the assistant's information source. Here's how it's processed by the system (primarily during the FastAPI backend startup):
### 1. Data Loading and Processing
*  File Reading: The app/main.py (specifically the load_and_process_knowledge_base function and the startup lifespan manager) reads the knowledge.json file.
*  Content Aggregation: For each item within categories like city_services, public_facilities, etc., the system constructs a comprehensive text string. This string includes:
   *  Title
   *  Category
   *  Content
   *  And other relevant fields like contact, office_hours, location, address, hours, phone, website, and emergency if they exist for that item.
*  Each piece of information is prefixed with its field name (e.g., "Title: ...", "Contact: ...") for better readability and context for the LLM.
  
### 2. Chunking
*  Purpose: To break down potentially long Document objects into smaller, manageable pieces. This is important for:
*  Fitting within the token limits of embedding models.
  *  Method: The RecursiveCharacterTextSplitter from LangChain is used.

### 3. Embedding and Indexing
*  Embedding Model: A Hugging Face sentence transformer model (e.g., sentence-transformers/all-MiniLM-L6-v2) is used. This model converts each text chunk into a dense numerical vector (embedding) that captures its semantic meaning.
### 4. Vector Store (FAISS):
*  The generated embeddings for all chunks are stored in a FAISS (Facebook AI Similarity Search) index.

This processed and indexed knowledge base forms the foundation for the RAG pipeline, enabling the assistant to retrieve relevant information before generating an answer.

## API Endpoints
You can also interact with the FastAPI backend directly using tools like curl, Postman, or Insomnia. The base URL is http://localhost:8000.
For demonstration purpose I have used Curl
```bash
 curl ST "http://localhost:8000/query" \
> -H "Content-Type: application/json" \
> -d '{
>     "query": "How do I apply for a building permit?"
> }'
```
You can change the parameter to 'search' and 'health' to test different APIs

Below is a summary of the key endpoints:

### GET /health
Description: Checks the health of the service.
Request: None
Response (200 OK):
```json
{
  "status": "OK: RAG components should be loaded if no errors logged during startup."
}
```
### POST /search
Description: Performs a direct semantic search on the vector store and returns relevant documents.
Request Body:
```json
{
  "query": "your search query",
  "k": 3 // Optional: number of documents to retrieve (default: 3)
}
```
Response (200 OK):
```json
{
  "query": "your search query",
  "retrieved_documents": [
    {
      "page_content": "Text content of the document...",
      "metadata": { "id": "CS001", "title": "...", "category": "..." }
    }
    // ... more documents
  ]
}
```

Error Responses:
503 Service Unavailable: If the vector store is not initialized.
500 Internal Server Error: If an error occurs during the search.

### POST /query
Description: Processes a query using the full RAG pipeline (retrieval + LLM generation).
Request Body:
```json
{
  "query": "your question for the assistant"
}
```
Response (200 OK):
```json
{
  "query": "your question for the assistant",
  "answer": "The assistant's generated answer...",
  "source_documents": [
    {
      "page_content": "Text content of the source document...",
      "metadata": { "id": "CS001", "title": "...", "category": "..." }
    }
    // ... more source documents
  ]
}
```
Error Responses:
503 Service Unavailable: If the RAG chain is not initialized (e.g., LLM or vector store missing).
500 Internal Server Error: If an error occurs during RAG processing.

### POST /crew-ask
Description: Processes a task description using the CrewAI multi-agent system (if implemented and enabled).
Request Body:
```json
{
  "task_description": "Describe your complex task or question for the crew"
}
```
Response (200 OK):
```json
{
  "result": "Output from the CrewAI kickoff process..." // Structure can vary
}
```
Error Responses:
503 Service Unavailable: If GOOGLE_API_KEY is not set for CrewAI agents.
500 Internal Server Error: If an error occurs during CrewAI processing.
