import streamlit as st
import os
from rag_system import RAGSystem
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System with Gemini AI",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def initialize_rag_system():
    """Initialize the RAG system with Gemini API key."""
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        st.error("GEMINI_API_KEY environment variable not set!")
        st.info("Please set your Gemini API key in a .env file or environment variable.")
        return None
    
    try:
        return RAGSystem(gemini_api_key)
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

def load_and_process_documents(rag_system, file_path, text_field, metadata_fields):
    """Load and process documents from JSON file."""
    try:
        # Load documents
        documents = rag_system.load_json_dataset(file_path, text_field, metadata_fields)
        
        # Process documents into chunks
        processed_chunks = rag_system.process_documents(documents, chunk_size=500, overlap=100)
        
        # Add to vector store
        rag_system.add_documents_to_vectorstore(processed_chunks)
        
        return len(documents), len(processed_chunks)
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return 0, 0

def main():
    st.title("🤖 RAG System with Gemini AI")
    st.markdown("Retrieval-Augmented Generation system using custom JSON datasets")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Initialize RAG system
        if st.button("Initialize RAG System"):
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = initialize_rag_system()
                if st.session_state.rag_system:
                    st.success("RAG system initialized successfully!")
        
        # Document upload section
        st.header("Load Documents")
        
        # Option to use example dataset
        if st.checkbox("Use Example Dataset"):
            if st.session_state.rag_system and st.button("Load Example Dataset"):
                with st.spinner("Loading example dataset..."):
                    num_docs, num_chunks = load_and_process_documents(
                        st.session_state.rag_system,
                        "example_dataset.json",
                        "content",
                        ["title", "category", "author", "date"]
                    )
                    if num_docs > 0:
                        st.session_state.documents_loaded = True
                        st.success(f"Loaded {num_docs} documents ({num_chunks} chunks)")
        
        # Custom JSON file upload
        uploaded_file = st.file_uploader(
            "Upload JSON Dataset",
            type=['json'],
            help="Upload a JSON file with your documents"
        )
        
        if uploaded_file and st.session_state.rag_system:
            # Parse uploaded file
            try:
                data = json.load(uploaded_file)
                st.write(f"File contains {len(data)} items")
                
                # Get field names from first item
                if data and isinstance(data[0], dict):
                    fields = list(data[0].keys())
                    text_field = st.selectbox("Select text field", fields)
                    metadata_fields = st.multiselect("Select metadata fields", fields)
                    
                    if st.button("Load Custom Dataset"):
                        with st.spinner("Loading custom dataset..."):
                            # Save uploaded file temporarily
                            with open("temp_dataset.json", "w") as f:
                                json.dump(data, f)
                            
                            num_docs, num_chunks = load_and_process_documents(
                                st.session_state.rag_system,
                                "temp_dataset.json",
                                text_field,
                                metadata_fields
                            )
                            if num_docs > 0:
                                st.session_state.documents_loaded = True
                                st.success(f"Loaded {num_docs} documents ({num_chunks} chunks)")
                                
                            # Clean up temp file
                            os.remove("temp_dataset.json")
            except Exception as e:
                st.error(f"Error parsing JSON file: {e}")
        
        # System stats
        if st.session_state.rag_system:
            st.header("System Stats")
            stats = st.session_state.rag_system.get_collection_stats()
            st.metric("Total Documents", stats['total_documents'])
    
    # Main content area
    if not st.session_state.rag_system:
        st.info("Please initialize the RAG system using the sidebar.")
        return
    
    if not st.session_state.documents_loaded:
        st.info("Please load documents using the sidebar.")
        return
    
    # Query interface
    st.header("Ask Questions")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning?",
        key="query_input"
    )
    
    # Query parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of documents to retrieve", 1, 10, 3)
    with col2:
        max_tokens = st.slider("Max response tokens", 100, 2000, 1000)
    
    # Process query
    if query and st.button("Ask Question"):
        with st.spinner("Processing query..."):
            try:
                result = st.session_state.rag_system.query(query, top_k, max_tokens)
                
                # Display response
                st.subheader("Response")
                st.write(result['response'])
                
                # Display retrieved documents
                st.subheader(f"Retrieved Documents ({result['num_documents_retrieved']})")
                
                for i, doc in enumerate(result['retrieved_documents'], 1):
                    with st.expander(f"Document {i} - {doc['metadata'].get('title', 'Untitled')}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Content:**")
                            st.write(doc['text'])
                        
                        with col2:
                            st.write("**Metadata:**")
                            for key, value in doc['metadata'].items():
                                st.write(f"**{key}:** {value}")
                            
                            relevance_score = 1 - doc['distance']
                            st.write(f"**Relevance:** {relevance_score:.3f}")
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    # Example queries
    st.header("Example Queries")
    example_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is RAG and how does it work?",
        "Tell me about vector databases",
        "What is Gemini AI?",
        "How do sentence transformers work?"
    ]
    
    cols = st.columns(3)
    for i, example_query in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(example_query, key=f"example_{i}"):
                st.session_state.query_input = example_query
                st.rerun()

if __name__ == "__main__":
    main()