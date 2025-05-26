
import streamlit as st
import requests # To make API calls
import json 

# --- Configuration ---
FASTAPI_URL = "http://localhost:8000" # URL of your FastAPI backend

# --- Page Configuration ---
st.set_page_config(
    page_title="Metro City Assistant",
    page_icon="🏙️",
    layout="wide"
)

# --- Main Application UI ---
st.title("🏙️ Metro City Information Assistant")
st.markdown("Ask me anything about Metro City services, facilities, policies, and more!")

# --- Sidebar for different modes or advanced options ---
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Chat with Assistant (/query)", "Direct Vector Search (/search)", "Ask the Crew (/crew-ask)"]
)

# --- Input area for user query ---
user_query = st.text_input("What would you like to know?", key="query_input")

if app_mode == "Chat with Assistant (/query)":
    st.subheader("Chat with the Assistant")
    if st.button("Ask Assistant", key="ask_button"):
        if user_query:
            with st.spinner("Thinking..."):
                try:
                    payload = {"query": user_query}
                    response = requests.post(f"{FASTAPI_URL}/query", json=payload)
                    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                    
                    rag_response = response.json()
                    
                    st.markdown("#### Assistant's Answer:")
                    st.info(rag_response.get("answer", "No answer received."))
                    
                    if rag_response.get("source_documents"):
                        st.markdown("---")
                        st.markdown("#### Retrieved Source Documents:")
                        for i, doc in enumerate(rag_response["source_documents"]):
                            with st.expander(f"Source {i+1}: {doc.get('metadata', {}).get('title', 'Unknown Title')}"):
                                st.caption(f"ID: {doc.get('metadata', {}).get('id', 'N/A')}, Category: {doc.get('metadata', {}).get('category', 'N/A')}")
                                st.markdown(doc.get("page_content", "No content available."))
                
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: Could not connect to the backend or an API error occurred. {e}")
                    try:
                        error_detail = e.response.json().get("detail", "No detail provided.")
                        st.error(f"Backend Error Detail: {error_detail}")
                    except:
                        pass # If parsing response JSON fails
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter a question.")

elif app_mode == "Direct Vector Search (/search)":
    st.subheader("Direct Vector Search")
    k_value = st.sidebar.number_input("Number of documents to retrieve (k)", min_value=1, max_value=10, value=3, key="k_search")

    if st.button("Search Documents", key="search_button"):
        if user_query:
            with st.spinner("Searching..."):
                try:
                    payload = {"query": user_query, "k": k_value}
                    response = requests.post(f"{FASTAPI_URL}/search", json=payload)
                    response.raise_for_status()

                    search_results = response.json()

                    st.markdown(f"#### Search Results for: \"{search_results.get('query')}\"")
                    if search_results.get("retrieved_documents"):
                        for i, doc in enumerate(search_results["retrieved_documents"]):
                            st.markdown(f"---")
                            st.markdown(f"**Document {i+1}: {doc.get('metadata', {}).get('title', 'Unknown Title')}**")
                            st.caption(f"ID: {doc.get('metadata', {}).get('id', 'N/A')}, Category: {doc.get('metadata', {}).get('category', 'N/A')}")
                            with st.expander("View Content", expanded=False):
                                st.markdown(doc.get("page_content", "No content available."))
                    else:
                        st.info("No documents found for your query.")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: Could not connect to the backend or an API error occurred. {e}")
                    try:
                        error_detail = e.response.json().get("detail", "No detail provided.")
                        st.error(f"Backend Error Detail: {error_detail}")
                    except:
                        pass
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter a search query.")
elif app_mode == "Ask the Crew (/crew-ask)":
    st.subheader("Ask the CrewAI Agents")
    # user_query is already defined above
    if st.button("Submit to Crew", key="crew_ask_button"):
        if user_query:
            with st.spinner("Crew is working on it... This may take a bit longer."):
                try:
                    payload = {"task_description": user_query}
                    response = requests.post(f"{FASTAPI_URL}/crew-ask", json=payload)
                    response.raise_for_status()
                    
                    crew_response = response.json()
                    
                    st.markdown("#### Crew's Response:")
                    # The output from crew.kickoff() can be a string or a more complex object
                    # depending on how tasks are structured and their expected_output.
                    # For now, just display the raw result.
                    st.json(crew_response.get("result", "No result from crew.")) 
                
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: {e}")
                    try:
                        st.error(f"Backend Error Detail: {e.response.json().get('detail')}")
                    except: pass
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please describe the task for the crew.")

# --- Health Check ---
st.sidebar.markdown("---")
if st.sidebar.button("Check Backend Health", key="health_check_button"):
    try:
        response = requests.get(f"{FASTAPI_URL}/health")
        response.raise_for_status()
        health_status = response.json().get("status", "Unknown status")
        st.sidebar.success(f"Backend Health: {health_status}")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Backend Unreachable or Error: {e}")
    except Exception as e:
        st.sidebar.error(f"Error checking health: {e}")

st.markdown("---")
st.caption("Powered by Streamlit and FastAPI | Smart City Assistant v1.0")