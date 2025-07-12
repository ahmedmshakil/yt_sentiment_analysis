import json
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pandas as pd
from typing import List, Dict, Any, Optional
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-pro"):
        """
        Initialize the RAG system with Gemini AI and ChromaDB for vector storage.
        
        Args:
            gemini_api_key: Google Gemini API key
            model_name: Gemini model to use (default: gemini-pro)
        """
        # Initialize Gemini AI
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(model_name)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def load_json_dataset(self, file_path: str, text_field: str = "content", 
                         metadata_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load documents from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            text_field: Field name containing the text content
            metadata_fields: List of fields to include as metadata
            
        Returns:
            List of documents with text and metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and text_field in item:
                doc = {
                    'id': str(i),
                    'text': item[text_field],
                    'metadata': {}
                }
                
                # Add metadata fields
                if metadata_fields:
                    for field in metadata_fields:
                        if field in item:
                            doc['metadata'][field] = item[field]
                
                documents.append(doc)
        
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]], 
                         chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Process documents by chunking them and preparing for embedding.
        
        Args:
            documents: List of documents
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens
            
        Returns:
            List of processed chunks
        """
        processed_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['text'], chunk_size, overlap)
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{j}"
                processed_chunks.append({
                    'id': chunk_id,
                    'text': chunk,
                    'metadata': doc['metadata'].copy(),
                    'original_doc_id': doc['id']
                })
        
        return processed_chunks
    
    def add_documents_to_vectorstore(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of processed document chunks
        """
        texts = [doc['text'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} document chunks to vector store")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: User query
            top_k: Number of top documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_docs
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                         max_tokens: int = 1000) -> str:
        """
        Generate a response using Gemini AI based on retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: Retrieved relevant documents
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated response
        """
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the user's question. 
        If the answer cannot be found in the context, say so clearly.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""
        
        # Generate response
        response = self.gemini_model.generate_content(prompt)
        
        return response.text
    
    def query(self, question: str, top_k: int = 5, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant documents and generate response.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary containing response and retrieved documents
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(question, top_k)
        
        # Generate response
        response = self.generate_response(question, retrieved_docs, max_tokens)
        
        return {
            'response': response,
            'retrieved_documents': retrieved_docs,
            'num_documents_retrieved': len(retrieved_docs)
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }