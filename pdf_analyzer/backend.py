import streamlit as st
import pymupdf
from streamlit_pdf_viewer import pdf_viewer
from openai import OpenAI
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
try:
    open_ai_api_key = st.secrets["OPEN_AI_KEY"]
except:
    open_ai_api_key = os.getenv('OPEN_AI_KEY')

def file_reader(uploaded_file):
    if uploaded_file: 
        binary_data = uploaded_file.getvalue()

        doc = pymupdf.open(stream=binary_data, filetype='pdf')
        text = ""
        for page in doc:
            text += page.get_text()

        doc.close()

        return binary_data, text
    
    else:
        return None, None

def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks for better context retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(texts, batch_size=100):
    """Get embeddings from OpenAI for a list of texts with batching"""
    client = OpenAI(api_key=open_ai_api_key)
    
    all_embeddings = []
    
    # Process in batches to avoid token limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings, dtype='float32')

def create_vector_store(pdf_name, text):
    """Create FAISS vector store for a PDF"""
    # Create chunks
    chunks = create_chunks(text)
    
    # Get embeddings for all chunks
    embeddings = get_embeddings(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # Dimension of embeddings (1536 for text-embedding-3-small)
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings)
    
    # Return index, chunks, and metadata
    vector_data = {
        'index': index,
        'chunks': chunks,
        'embeddings': embeddings
    }
    
    return vector_data, len(chunks)

def retrieve_relevant_chunks(vector_data, query, n_results=5):
    """Retrieve relevant chunks based on query using FAISS"""
    try:
        if not vector_data or 'index' not in vector_data:
            return []
        
        # Get embedding for the query
        query_embedding = get_embeddings([query])
        
        # Search FAISS index
        distances, indices = vector_data['index'].search(query_embedding, n_results)
        
        # Get the corresponding chunks
        relevant_chunks = [vector_data['chunks'][i] for i in indices[0] if i < len(vector_data['chunks'])]
        
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []

def doc_analysis(text, user_input, user_choice, vector_data=None, use_rag=True):
    client = OpenAI(api_key=open_ai_api_key)

    if user_choice.lower() == 'summary':
        user_prompt = 'Provide a comprehensive summary of this document'
        use_rag = False  # Use full text for summaries
    elif user_choice.lower() == 'complex_summary':
        user_prompt = 'Provide a detailed, complex analysis and summary of this document with technical insights and deep analysis'
        use_rag = False  # Use full text for summaries
    elif user_choice.lower() == 'simple_summary':
        user_prompt = 'Provide a very simple summary of this document using easy-to-understand language, as if explaining to a beginner'
        use_rag = False  # Use full text for summaries
    elif user_choice.lower() == 'analyze':
        user_prompt = user_input
        use_rag = True  # Use RAG for specific questions
    
    # Determine which text to use
    if use_rag and vector_data:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(vector_data, user_prompt, n_results=5)
        if relevant_chunks:
            context_text = "\n\n".join(relevant_chunks)
            context_info = f"Retrieved {len(relevant_chunks)} most relevant sections from the document."
        else:
            context_text = text
            context_info = "Using full document (RAG retrieval failed)."
    else:
        context_text = text
        context_info = "Using full document for summary."

    response = client.chat.completions.create(
        model='gpt-4o-mini',  
        messages=[
            {
                'role': 'system',
                'content': (
                    f"You are a professional yet friendly document analysis assistant. "
                    f"Analyze the document content and answer the user's question clearly and concisely. "
                    f"Document content: {context_text} " 
                    f"### Guidelines:\n"
                    f"1. Analyze the content in detail — summarize, extract facts, or explain meaning as needed.\n"
                    f"2. Use emojis **only in headings or subheadings** to make the structure more readable "
                    f"(e.g., '📄 Summary', '🔍 Key Insights', '💡 Answer').\n"
                    f"3. Keep the rest of the text clean and professional — no unnecessary emojis.\n"
                    f"4. Structure responses with short sections, bullet points, and clear formatting.\n"
                    f"5. When asked objective questions, present the answer first and then explain how you came to that conclusion by using facts and explanations."
                )
            },  
            {'role': 'user', 'content': user_prompt}
        ]
    )

    analysis = response.choices[0].message.content

    return analysis

