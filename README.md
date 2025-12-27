# Intelligent Document Search using RAG

A Retrieval-Augmented Generation (RAG) application that enables intelligent question-answering from PDF documents using state-of-the-art language models and vector search.

🔗 **[Live Demo](https://rag-pdf-app-tanvir.streamlit.app/)**

## Features

- 📄 Upload and process multiple PDF documents
- 🔍 Semantic search using FAISS vector database
- 🤖 Powered by Groq's Llama 3.1 model for fast inference
- 💬 Context-aware question answering
- 📊 View source documents for each answer
- ⚡ Real-time response time tracking

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.1-8B)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Framework**: LangChain

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-pdf-app.git
cd rag-pdf-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload one or more PDF files using the sidebar
2. Click "Create Vector Database" to process documents
3. Enter your question in the text input
4. View the answer along with source documents

## Configuration

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Model**: llama-3.1-8b-instant
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2

## Requirements
```
streamlit
langchain
langchain-groq
langchain-community
faiss-cpu
sentence-transformers
pypdf
python-dotenv
```

## License

MIT License

## Author

Tanvir Ahammed

Created with ❤️ using Streamlit, LangChain, Groq, and HuggingFace
