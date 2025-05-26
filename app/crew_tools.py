from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai

from crewai.tools import BaseTool # Import BaseTool

# --- Configuration (can be loaded from a config file or env vars) ---
FAISS_INDEX_PATH = "../faiss_index_city_info_hf" # Adjust as needed
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Or your preferred LLM

# --- Global RAG Components (Load once) ---
# This simplified loader assumes components are ready. In a real app,
# you'd ensure they are loaded during app startup (like in your FastAPI lifespan).
_vector_store = None
_llm = None
_rag_chain_for_tool = None # A specific RAG chain for direct answers if needed

def initialize_rag_components_for_crew():
    global _vector_store, _llm, _rag_chain_for_tool, hf_embeddings # Add hf_embeddings to global

    print("CrewAI Tools: Initializing RAG components...")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set. Required for LLM.")
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(FAISS_INDEX_PATH):
        _vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            hf_embeddings,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded for CrewAI tools.")
    else:
        # In a real scenario, you'd probably want to stop if the index isn't there,
        # or have a fallback to create it (which is more complex for a simple tool setup).
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. CrewAI tools relying on it will fail.")

    _llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0.3)
    print("Gemini LLM initialized for CrewAI tools.")

    # Optional: A simple RAG chain if a tool needs to generate a direct answer
    # based on retrieved context, rather than just returning raw documents.
    if _vector_store and _llm:
        prompt_template_str = """Based SOLELY on the following context, answer the question.
If the information is not in the context, say "I don't have enough information from the provided documents."
Context:
{context}
Question: {question}
Answer:"""
        PROMPT = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
        retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
        _rag_chain_for_tool = RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False # Usually tools return processed info, not raw docs for other agents
        )
        print("Simple RAG chain for tools initialized.")


# --- Define Custom Tools ---

class CityKnowledgeSearchTool(BaseTool):
    name: str = "City Knowledge Base Search"
    description: str = (
        "Searches the Metro City official knowledge base for information relevant to a query. "
        "Input should be a clear question or topic to search for. "
        "Returns retrieved documents or a summarized answer based on them."
    )

    def _run(self, query: str) -> str:
        if not _vector_store:
            return "Error: Vector store is not initialized."
        if not _llm: # Needed for summarization or direct RAG answer
             return "Error: LLM for tool is not initialized."
        if not _rag_chain_for_tool:
            # Fallback: Just return raw docs if full RAG chain for tool isn't set up
            # This is less ideal as agents prefer processed text.
            docs = _vector_store.similarity_search(query, k=3)
            if not docs:
                return "No relevant information found in the knowledge base for your query."
            # Combine content of retrieved docs
            combined_content = "\n\n---\n\n".join([f"Source Title: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}" for doc in docs])
            return f"Retrieved information:\n{combined_content}"

        # Use the simple RAG chain to get a summarized answer
        try:
            result = _rag_chain_for_tool.invoke({"query": query})
            return result.get('result', "No answer could be generated from the retrieved documents.")
        except Exception as e:
            return f"Error during knowledge search: {str(e)}"

# Call initialization when this module is loaded (or from FastAPI lifespan)
# For now, let's assume it's called before agents are created.
# initialize_rag_components_for_crew() # You'd call this from your FastAPI startup