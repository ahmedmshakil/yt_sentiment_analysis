# RAG System with Gemini AI

A complete Retrieval-Augmented Generation (RAG) system that uses Google's Gemini AI for text generation and ChromaDB for vector storage. This system can work with custom JSON datasets to provide accurate, context-aware responses.

## Features

- 🤖 **Gemini AI Integration**: Uses Google's latest Gemini Pro model for text generation
- 📚 **Custom JSON Datasets**: Load and process your own JSON-formatted documents
- 🔍 **Semantic Search**: Advanced document retrieval using sentence transformers
- 💾 **Vector Storage**: Efficient document storage with ChromaDB
- 🧩 **Smart Chunking**: Intelligent text chunking with overlap for better context
- 🌐 **Web Interface**: User-friendly Streamlit web application
- 📊 **Metadata Support**: Preserve and search document metadata
- ⚡ **Fast Retrieval**: Optimized vector similarity search

## Architecture

```
User Query → Embedding → Vector Search → Document Retrieval → Context + Query → Gemini AI → Response
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-system-gemini
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key**:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

## Usage

### 1. Command Line Interface

Run the main script for an interactive command-line experience:

```bash
python main.py
```

This will:
- Load the example dataset
- Process documents into chunks
- Add them to the vector store
- Start an interactive query session

### 2. Web Interface

Launch the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

The web interface provides:
- Easy document upload
- Interactive query interface
- Visual document retrieval results
- System statistics

### 3. Programmatic Usage

```python
from rag_system import RAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize RAG system
rag = RAGSystem(os.getenv('GEMINI_API_KEY'))

# Load your JSON dataset
documents = rag.load_json_dataset(
    file_path="your_dataset.json",
    text_field="content",
    metadata_fields=["title", "author", "category"]
)

# Process documents
chunks = rag.process_documents(documents)

# Add to vector store
rag.add_documents_to_vectorstore(chunks)

# Query the system
result = rag.query("What is machine learning?")
print(result['response'])
```

## JSON Dataset Format

Your JSON dataset should be an array of objects. Each object should contain:

```json
[
  {
    "content": "The main text content of the document...",
    "title": "Document Title",
    "author": "Author Name",
    "category": "Category",
    "date": "2024-01-01",
    "other_metadata": "Additional information"
  }
]
```

**Required fields:**
- `content`: The main text content (field name can be customized)

**Optional fields:**
- Any other fields will be preserved as metadata and can be used for filtering/searching

## Configuration

### RAG System Parameters

- **Chunk Size**: Maximum tokens per document chunk (default: 1000)
- **Overlap**: Number of overlapping tokens between chunks (default: 200)
- **Top K**: Number of documents to retrieve for each query (default: 5)
- **Max Tokens**: Maximum tokens for Gemini AI response (default: 1000)

### Embedding Model

The system uses `all-MiniLM-L6-v2` for generating embeddings. This model provides a good balance between performance and accuracy.

## Example Queries

Try these example queries with the provided dataset:

- "What is machine learning?"
- "How does deep learning work?"
- "What is RAG and how does it work?"
- "Tell me about vector databases"
- "What is Gemini AI?"
- "How do sentence transformers work?"

## File Structure

```
rag-system-gemini/
├── rag_system.py          # Main RAG system class
├── main.py               # Command-line interface
├── streamlit_app.py      # Web interface
├── example_dataset.json  # Sample dataset
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## Advanced Features

### Custom Embedding Models

You can modify the embedding model in `rag_system.py`:

```python
# Change the embedding model
self.embedding_model = SentenceTransformer('your-preferred-model')
```

### Custom Prompts

Modify the prompt template in the `generate_response` method:

```python
prompt = f"""Your custom prompt template here.
Context: {context}
Question: {query}
Answer:"""
```

### Metadata Filtering

Add metadata filtering to your queries by modifying the `retrieve_relevant_documents` method.

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Gemini API key is correctly set in the `.env` file
2. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Memory Issues**: Reduce chunk size for large datasets
4. **Slow Performance**: Consider using a smaller embedding model or reducing the number of retrieved documents

### Performance Optimization

- Use smaller chunk sizes for faster processing
- Reduce the number of retrieved documents (`top_k`)
- Consider using a smaller embedding model for faster inference
- Use GPU acceleration if available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Google Gemini AI](https://ai.google.dev/) for the language model
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Streamlit](https://streamlit.io/) for the web interface