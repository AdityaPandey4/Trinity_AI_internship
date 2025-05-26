
from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    k: int = 3 #number of documents to retrieve for /search and RAG

class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    retrieved_documents: List[DocumentResponse]

class RAGResponse(BaseModel):
    query: str
    answer: str
    source_documents: List[DocumentResponse]

class HealthResponse(BaseModel):
    status: str