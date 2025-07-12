#!/usr/bin/env python3
"""
Test script for the RAG system.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_rag_system():
    """Test the RAG system functionality."""
    
    # Check if API key is set
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please set your Gemini API key in a .env file")
        return False
    
    try:
        # Import RAG system
        from rag_system import RAGSystem
        
        print("✅ Successfully imported RAG system")
        
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag = RAGSystem(gemini_api_key)
        print("✅ RAG system initialized successfully")
        
        # Test document loading
        print("🔄 Testing document loading...")
        documents = rag.load_json_dataset(
            file_path="example_dataset.json",
            text_field="content",
            metadata_fields=["title", "category", "author", "date"]
        )
        print(f"✅ Loaded {len(documents)} documents")
        
        # Test document processing
        print("🔄 Testing document processing...")
        processed_chunks = rag.process_documents(documents, chunk_size=500, overlap=100)
        print(f"✅ Created {len(processed_chunks)} chunks")
        
        # Test vector store
        print("🔄 Testing vector store...")
        rag.add_documents_to_vectorstore(processed_chunks)
        
        # Get stats
        stats = rag.get_collection_stats()
        print(f"✅ Vector store contains {stats['total_documents']} documents")
        
        # Test query
        print("🔄 Testing query functionality...")
        test_query = "What is machine learning?"
        result = rag.query(test_query, top_k=2)
        
        print(f"✅ Query successful")
        print(f"   Response: {result['response'][:100]}...")
        print(f"   Retrieved {result['num_documents_retrieved']} documents")
        
        print("\n🎉 All tests passed! RAG system is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install all dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'google.generativeai',
        'sentence_transformers',
        'chromadb',
        'pandas',
        'numpy',
        'python-dotenv',
        'streamlit',
        'tiktoken'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing dependencies: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed")
        return True

def main():
    """Run all tests."""
    print("🧪 RAG System Test Suite")
    print("=" * 40)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    if not deps_ok:
        sys.exit(1)
    
    print("\n" + "=" * 40)
    
    # Test RAG system
    rag_ok = test_rag_system()
    
    if rag_ok:
        print("\n🎉 All tests passed! Your RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py' for command-line interface")
        print("2. Run 'streamlit run streamlit_app.py' for web interface")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()