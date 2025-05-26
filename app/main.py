from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager # For lifespan events
import os
import json

from crewai import Agent, Task, Crew, Process,LLM
from .crew_tools import CityKnowledgeSearchTool, initialize_rag_components_for_crew
# Langchain and RAG components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai # For configuring API key

from pydantic import BaseModel
class CrewTaskRequest(BaseModel):
    task_description: str
# Pydantic models from models.py
from .models import QueryRequest, SearchResponse, RAGResponse, HealthResponse, DocumentResponse

# --- Configuration ---
KNOWLEDGE_BASE_PATH = "./knowledge.json"  
FAISS_INDEX_PATH = "./faiss_index_city_info_hf" 
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # or your preferred Gemini model
os.environ['GOOGLE_API_KEY'] = 'AIzaSyABvQiCcX_uM_e_EyBkUG1wnEg6pa1sfow'
os.environ['GEMINI_API_KEY'] = 'AIzaSyABvQiCcX_uM_e_EyBkUG1wnEg6pa1sfow'
# Global variables for RAG components (will be initialized on startup)
vector_store = None
rag_chain = None
llm = None
hf_embeddings = None 

# --- Helper function to load and process knowledge base  ---
def load_and_process_knowledge_base(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_docs_content = []
    for category_key, items in data["knowledge_base"].items():
        if category_key == "test_queries":
            continue
        for item in items:
            page_content = f"Title: {item.get('title', '')}\n"
            page_content += f"Category: {item.get('category', '')}\n"
            page_content += f"Content: {item.get('content', '')}\n"
            if 'contact' in item: page_content += f"Contact: {item.get('contact')}\n"
            if 'office_hours' in item: page_content += f"Office Hours: {item.get('office_hours')}\n"
            if 'location' in item: page_content += f"Location: {item.get('location')}\n"
            if 'address' in item: page_content += f"Address: {item.get('address')}\n"
            if 'hours' in item: page_content += f"Hours: {item.get('hours')}\n"
            if 'phone' in item: page_content += f"Phone: {item.get('phone')}\n"
            if 'website' in item: page_content += f"Website: {item.get('website')}\n"
            if 'emergency' in item: page_content += f"Emergency: {item.get('emergency')}\n"
            metadata = {"id": item.get("id"), "title": item.get("title"), "category": item.get("category")}
            all_docs_content.append(Document(page_content=page_content.strip(), metadata=metadata))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_documents = text_splitter.split_documents(all_docs_content)
    return chunked_documents

# --- Lifespan event for FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Application startup: Initializing RAG pipeline...")
    global vector_store, rag_chain, llm, hf_embeddings

    # 1. Initialize Embeddings
    print(f"Loading Hugging Face embedding model: {HF_EMBEDDING_MODEL}")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Load or Create FAISS Vector Store
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        try:
            
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                hf_embeddings,
                allow_dangerous_deserialization=True # Add this if you encounter issues loading
            )
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Will try to re-create.")
            vector_store = None # Ensure it's None so it gets recreated

    if not vector_store:
        print(f"FAISS index not found or failed to load. Creating new index from {KNOWLEDGE_BASE_PATH}...")
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            raise FileNotFoundError(f"Knowledge base file not found at {KNOWLEDGE_BASE_PATH}. Cannot create FAISS index.")
        
        documents_for_faiss = load_and_process_knowledge_base(KNOWLEDGE_BASE_PATH)
        if not documents_for_faiss:
            raise ValueError("No documents processed from knowledge base. Cannot create FAISS index.")
            
        vector_store = FAISS.from_documents(documents_for_faiss, hf_embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index created and saved to {FAISS_INDEX_PATH}.")

    # 3. Initialize LLM (Gemini)
    print(f"Initializing Gemini LLM: {GEMINI_MODEL_NAME}")
    try:
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print("WARNING: GOOGLE_API_KEY environment variable not set. Gemini LLM may not work.")
        
        # Configure Gemini API key 
        genai.configure(api_key=google_api_key) 
        
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.3,
            # convert_system_message_to_human=True # If needed
        )
        print("Gemini LLM initialized.")
    except Exception as e:
        print(f"FATAL: Could not initialize Gemini LLM: {e}")
        print("The /query endpoint will not work. Please check your GOOGLE_API_KEY and Gemini setup.")
        llm = None # Ensure llm is None if initialization fails

    # 4. Create RAG Chain (if LLM is available)
    if llm and vector_store:
        prompt_template_str = """You are a Smart City Information Assistant for Metro City.
Your goal is to provide helpful, accurate, and concise answers to citizen's questions based ONLY on the provided context.
If the information is not in the context, say "I don't have information on that topic based on the provided documents." Do not make up answers.

Context:
{context}

Question: {question}

Helpful Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Default k for RAG chain
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("RAG chain created.")
    else:
        print("RAG chain not created due to missing LLM or Vector Store.")
    
    try:
        print("FastAPI Lifespan: Initializing components for CrewAI tools...")
        initialize_rag_components_for_crew() # This will set up _vector_store, _llm for tools
        print("CrewAI components initialized via lifespan.")
    except Exception as e:
        print(f"ERROR initializing CrewAI components: {e}")
        # Decide how to handle this: maybe disable CrewAI endpoint or log critical error


    yield
    # Code to run on shutdown (cleanup, if any)
    print("Application shutdown.")

# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Checks the health of the service."""
    return HealthResponse(status="OK: RAG components should be loaded if no errors above.")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: QueryRequest):
    """
    Performs a semantic search on the vector store and returns relevant documents.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized. Service may be starting up or encountered an error.")
    if not hf_embeddings: # Ensure embeddings are loaded for direct search
        raise HTTPException(status_code=503, detail="Embedding model not initialized.")

    print(f"Search request: query='{request.query}', k={request.k}")
    try:
        # Directly use the vector_store's similarity search
        retrieved_docs = vector_store.similarity_search(request.query, k=request.k)
        
        # Format for response
        response_docs = [
            DocumentResponse(page_content=doc.page_content, metadata=doc.metadata)
            for doc in retrieved_docs
        ]
        return SearchResponse(query=request.query, retrieved_documents=response_docs)
    except Exception as e:
        print(f"Error during /search: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


@app.post("/query", response_model=RAGResponse)
async def process_query(request: QueryRequest):
    """
    Processes a query using the full RAG pipeline (retrieval + LLM generation).
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized. LLM or Vector Store might be missing or encountered an error during startup.")
    
    print(f"RAG query request: query='{request.query}'")
    try:
        
        
        result = rag_chain.invoke({"query": request.query}) 

        response_docs = [
            DocumentResponse(page_content=doc.page_content, metadata=doc.metadata)
            for doc in result.get('source_documents', [])
        ]
        
        return RAGResponse(
            query=request.query,
            answer=result.get('result', "No answer generated."),
            source_documents=response_docs
        )
    except Exception as e:
        print(f"Error during /query: {e}")
        
        raise HTTPException(status_code=500, detail=f"Error processing RAG query: {str(e)}")

@app.post("/crew-ask", response_model=dict) # Using dict for flexible CrewAI output for now
async def crew_process_task(request: CrewTaskRequest):
    """
    Processes a task using a CrewAI setup.
    """
    if not os.environ.get("GOOGLE_API_KEY"): # CrewAI agents use LLMs directly
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY not set. CrewAI agents cannot function.")

    try:
        # 1. Define Tools
        knowledge_tool = CityKnowledgeSearchTool()

        # 2. Define Agents
        # Using the same LLM for all agents for simplicity, but you can customize
        agent_llm = LLM(model='gemini/gemini-1.5-flash-latest')
        retriever_agent = Agent(
            role='Information Retriever Specialist',
            goal=f'Search the Metro City knowledge base for information related to: {request.task_description}',
            backstory=(
                "You are an expert in navigating Metro City's digital archives and official documents. "
                "Your primary function is to find the most relevant pieces of information based on a specific query. "
                "You pass this raw or summarized information to other agents."
            ),
            tools=[knowledge_tool],
            llm=agent_llm,
            verbose=True, 
            allow_delegation=False 
        )

        policy_expert_agent = Agent(
            role='Policy Interpretation Expert',
            goal=f'Analyze information about city policies, ordinances, or regulations related to: {request.task_description}, and provide a clear explanation.',
            backstory=(
                "You are a specialist in Metro City\'s legal and regulatory framework. "
                "You take factual information and explain its implications regarding policies. "
                "You do not search for information yourself; you rely on what is provided to you by the Information Retriever."
            ),
            llm=agent_llm,
            verbose=True,
            allow_delegation=False 
        )

        service_coordinator_agent = Agent(
            role='City Service Navigator',
            goal=f'Provide step-by-step guidance, contact details, or necessary forms related to city services for: {request.task_description}.',
            backstory=(
                "You are a friendly and efficient guide for Metro City\'s services. "
                "You take information about services (like permits, utilities, waste management) and structure it into actionable advice for citizens. "
                "You rely on the Information Retriever for base information."
            ),
            llm=agent_llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Chief Orchestrator Agent (Optional, but good for more complex queries)
        
        retrieval_task = Task(
            description=(
                f"Find all relevant information from the Metro City knowledge base concerning: '{request.task_description}'. "
                "Focus on official procedures, requirements, contact details, and policy statements."
            ),
            expected_output=(
                "A concise summary of the retrieved information, including key facts, figures, and document titles or IDs if available. "
                "This output will be used by other expert agents."
            ),
            agent=retriever_agent
        )

        
        policy_analysis_task = Task(
            description=(
                "Based on the information retrieved about '{request.task_description}', analyze any relevant city policies, "
                "ordinances, or regulations. Explain them clearly. If no policy information is present in the retrieved context, state that."
            ),
            expected_output=(
                "A clear explanation of relevant policies or a statement if no policy information was found in the provided context. "
                "This output should be understandable by a citizen."
            ),
            agent=policy_expert_agent,
            context=[retrieval_task] # Depends on the output of the retrieval_task
        )

        # Task for the service coordinator
        service_guidance_task = Task(
            description=(
                "Using the retrieved information about '{request.task_description}', provide practical guidance. "
                "This might include step-by-step instructions for a service, necessary forms, office locations, or contact numbers. "
                "If the query is not about a direct city service, state that clearly."
            ),
            expected_output=(
                "Actionable, step-by-step guidance for the citizen, or relevant contact information. "
                "If not a service-related query based on context, indicate that."
            ),
            agent=service_coordinator_agent,
            context=[retrieval_task] # Also depends on the output of the retrieval_task
        )

        # 4. Create and Run the Crew
        
        final_tasks_to_run = [retrieval_task]
        query_lower = request.task_description.lower()

        if "policy" in query_lower or "ordinance" in query_lower or "regulation" in query_lower or "law" in query_lower:
            final_tasks_to_run.append(policy_analysis_task)
        elif "service" in query_lower or "permit" in query_lower or "license" in query_lower or "collection" in query_lower or "utility" in query_lower or "how do i" in query_lower:
            final_tasks_to_run.append(service_guidance_task)
        else:
            # If unsure, maybe default to a general info task or just retrieval output
            final_tasks_to_run.append(policy_analysis_task)


        city_crew = Crew(
            agents=[retriever_agent, policy_expert_agent, service_coordinator_agent],
            tasks=final_tasks_to_run, # Run selected tasks
            process=Process.sequential, # For this simple case, sequential is fine.
                                        # Hierarchical for manager/worker crews.
            verbose=True # 
        )

        crew_result = city_crew.kickoff()
        
        return {"result": crew_result}

    except Exception as e:
        print(f"Error during CrewAI processing: {e}")
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing task with CrewAI: {str(e)}")

# To run the app (from the directory containing the 'app' folder):
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

