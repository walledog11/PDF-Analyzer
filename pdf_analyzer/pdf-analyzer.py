import streamlit as st
from streamlit import session_state as ss
import pymupdf
from openai import OpenAI
from streamlit_pdf_viewer import pdf_viewer
from backend import file_reader, doc_analysis, create_vector_store
from time import sleep
from datetime import datetime

# Initialize session state variables
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

if 'pdf_library' not in ss:
    ss.pdf_library = []

if 'current_pdf_name' not in ss:
    ss.current_pdf_name = None

if 'vector_data' not in ss:
    ss.vector_data = None

st.set_page_config(
    page_title="PDF Analyzer",
    page_icon="📄",
    layout="wide"
)
st.title('PDF Analyzer')
st.markdown('Upload any PDF and get a summary')

# Sidebar - Library
with st.sidebar:
    st.header("📚 Library")
    st.markdown("---")
    
    # Display saved PDFs
    if ss.pdf_library:
        st.subheader("Saved PDFs")
        for idx, pdf_item in enumerate(ss.pdf_library):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                if st.button(f"📄 {pdf_item['name']}", key=f"load_{idx}", use_container_width=True):
                    # Load the selected PDF from library
                    ss.pdf_ref = pdf_item['name']
                    ss.binary_data = pdf_item['binary_data']
                    ss.pdf_text = pdf_item['text']
                    ss.current_pdf_name = pdf_item['name']
                    ss.vector_data = pdf_item.get('vector_data')
                    st.rerun()
            with col_b:
                if st.button("🗑️", key=f"delete_{idx}"):
                    ss.pdf_library.pop(idx)
                    st.rerun()
        
        st.markdown("---")
        st.caption(f"Total PDFs: {len(ss.pdf_library)}")
    else:
        st.info("No PDFs saved yet. Upload and save a PDF to build your library!")

uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'], key='pdf')

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file:
        ss.pdf_ref = uploaded_file
        ss.current_pdf_name = uploaded_file.name

        # Only process if it's a new file or not yet processed
        if 'pdf_text' not in ss or ss.get('last_processed_file') != uploaded_file.name:
            with st.spinner('Processing PDF...'):
                try: 
                    binary_data, text = file_reader(uploaded_file)
                    ss.pdf_text = text
                    ss.binary_data = binary_data
                    ss.last_processed_file = uploaded_file.name
                    
                except Exception as e:
                    st.error(e)
            
            # Create vector embeddings only once per file
            with st.spinner('Creating vector embeddings for semantic search...'):
                try:
                    vector_data, num_chunks = create_vector_store(ss.current_pdf_name, ss.pdf_text)
                    ss.vector_data = vector_data
                    st.success(f"✅ Created {num_chunks} chunks with embeddings for efficient retrieval")
                except Exception as e:
                    st.error(f"Error creating embeddings: {e}")
                    ss.vector_data = None

    if ss.pdf_ref and 'binary_data' in ss:
        # Show save to library button
        if ss.current_pdf_name:
            # Check if PDF is already in library
            is_in_library = any(pdf['name'] == ss.current_pdf_name for pdf in ss.pdf_library)
            
            if not is_in_library:
                if st.button("💾 Save to Library", use_container_width=True):
                    ss.pdf_library.append({
                        'name': ss.current_pdf_name,
                        'binary_data': ss.binary_data,
                        'text': ss.pdf_text,
                        'vector_data': ss.vector_data,
                        'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success(f"✅ '{ss.current_pdf_name}' saved to library!")
                    st.rerun()
            else:
                st.info(f"📚 '{ss.current_pdf_name}' is already in your library")
        
        pdf_viewer(input=ss.binary_data, 
                width=700,
                height=1000,
                render_text=True,
                )

    else:
        st.info("Please upload a valid file")

with col2:
    if ss.pdf_ref and 'pdf_text' in ss:
        with st.container():
            gen_summarize = st.button("General Summary", use_container_width=True)
            complex_summarize = st.button("Complex Summary", use_container_width=True)
            simple_summarize = st.button("Simplest Language Summary", use_container_width=True)

        with st.form("Analysis"):
            user_input = st.text_input('What questions do you have about this document?')
            submitted = st.form_submit_button("Submit", use_container_width=True)

        if gen_summarize:
            with st.spinner("Summarizing document..."):
                try:
                    summary = doc_analysis(ss.pdf_text, "", user_choice='summary', vector_data=ss.vector_data)
                    with st.container(height=1000):
                        st.markdown(summary)
                except Exception as e:
                    st.error(f"AI Error: {e}")

        if complex_summarize:
            with st.spinner("Summarizing document..."):
                try:
                    summary = doc_analysis(ss.pdf_text, "", user_choice='complex_summary', vector_data=ss.vector_data)
                    with st.container(height=1000):
                        st.markdown(summary)
                except Exception as e:
                    st.error(f"AI Error: {e}")

        if simple_summarize:
            with st.spinner("Summarizing document..."):
                try:
                    summary = doc_analysis(ss.pdf_text, "", user_choice='simple_summary', vector_data=ss.vector_data)
                    with st.container(height=1000):
                        st.markdown(summary)
                except Exception as e:
                    st.error(f"AI Error: {e}")

        if submitted and user_input:  
            with st.spinner("Analyzing document with RAG (retrieving relevant sections)..."):
                try: 
                    analysis = doc_analysis(ss.pdf_text, user_input, user_choice='analyze', vector_data=ss.vector_data)
                    with st.container(height=1000):
                        st.markdown(analysis)
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.info("Please upload a PDF file")

