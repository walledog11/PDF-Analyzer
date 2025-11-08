# PDF Analyzer

A Streamlit web application that allows users to upload PDF files and analyze them using OpenAI's GPT models with RAG (Retrieval Augmented Generation) for efficient document processing.

## Features

- 📄 Upload and view PDF files
- 🔍 Extract text from PDFs with intelligent chunking
- 🤖 AI-powered document summarization (general, complex, or simple)
- 💬 Ask specific questions about document content
- 📚 Sidebar library to save and manage multiple PDFs
- 🎯 **RAG Implementation**: Vector embeddings with FAISS for semantic search
- ⚡ Efficient context retrieval - only relevant sections sent to AI
- ☁️ Streamlit Cloud compatible (uses session-based storage)

## RAG Architecture

The app uses **FAISS (Facebook AI Similarity Search)** for vector storage:
- **Text Chunking**: Documents split into 1000-character chunks with 200-char overlap
- **Embeddings**: OpenAI's `text-embedding-3-small` model
- **Storage**: In-memory FAISS indexes stored in Streamlit session state
- **Retrieval**: Top 5 most relevant chunks retrieved for each query
- **Cost Savings**: Only sends relevant context to GPT, reducing token usage

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pdf-analyzer.git
cd pdf-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your OpenAI API key:
```
OPEN_AI_KEY=your-openai-api-key-here
```

4. Run the application:
```bash
streamlit run pdf-analyzer.py
```

## Usage

1. Upload a PDF file using the file uploader
2. Wait for the PDF to be processed
3. Click "Summarize Document" for a full summary
4. Or ask specific questions about the document

## Requirements

- Python 3.7+
- Streamlit
- OpenAI API key
- PyMuPDF
- python-dotenv
- streamlit-pdf-viewer

## License

MIT License