#!/usr/bin/env python3
"""
Main script to demonstrate the RAG system with Gemini AI.
"""

import os
from rag_system import RAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get Gemini API key from environment variable
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key in a .env file or environment variable.")
        return
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem(gemini_api_key)
    
    # Load and process documents
    print("Loading documents from JSON dataset...")
    documents = rag.load_json_dataset(
        file_path="example_dataset.json",
        text_field="content",
        metadata_fields=["title", "category", "author", "date"]
    )
    
    print(f"Loaded {len(documents)} documents")
    
    # Process documents into chunks
    print("Processing documents into chunks...")
    processed_chunks = rag.process_documents(documents, chunk_size=500, overlap=100)
    
    print(f"Created {len(processed_chunks)} chunks")
    
    # Add documents to vector store
    print("Adding documents to vector store...")
    rag.add_documents_to_vectorstore(processed_chunks)
    
    # Get collection statistics
    stats = rag.get_collection_stats()
    print(f"Vector store contains {stats['total_documents']} documents")
    
    # Interactive query loop
    print("\n" + "="*50)
    print("RAG System Ready! Ask questions about the documents.")
    print("Type 'quit' to exit.")
    print("="*50)
    
    while True:
        try:
            # Get user query
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query
            print("Processing query...")
            result = rag.query(query, top_k=3)
            
            # Display results
            print("\n" + "-"*50)
            print("RESPONSE:")
            print(result['response'])
            print("\n" + "-"*50)
            print(f"Retrieved {result['num_documents_retrieved']} relevant documents:")
            
            for i, doc in enumerate(result['retrieved_documents'], 1):
                print(f"\nDocument {i}:")
                print(f"Title: {doc['metadata'].get('title', 'N/A')}")
                print(f"Category: {doc['metadata'].get('category', 'N/A')}")
                print(f"Author: {doc['metadata'].get('author', 'N/A')}")
                print(f"Relevance Score: {1 - doc['distance']:.3f}")
                print(f"Content: {doc['text'][:200]}...")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()