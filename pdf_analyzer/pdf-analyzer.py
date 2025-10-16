import streamlit as st
from streamlit import session_state as ss
import pymupdf
from openai import OpenAI
from streamlit_pdf_viewer import pdf_viewer
from backend import file_reader, doc_analysis
from time import sleep

if 'pdf_ref' not in ss:
    ss.pdf_ref = None

st.set_page_config(
    page_title="PDF Analyzer",
    page_icon="📄",
    layout="wide"
)
st.title('PDF Analyzer')
st.markdown('Upload any PDF and get a summary')

uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'], key='pdf')

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file:
        ss.pdf_ref = uploaded_file

        with st.spinner('Processing PDF...'):
            try: 
                binary_data, text = file_reader(uploaded_file)
                ss.pdf_text = text
                ss.binary_data = binary_data
                
            except Exception as e:
                st.error(e)

    if uploaded_file and 'binary_data' in ss:
        pdf_viewer(input=ss.binary_data, 
                width=700,
                height=1000,
                render_text=True,
                )

    else:
        st.info("Please upload a valid file")

with col2:
    if uploaded_file and 'pdf_text' in ss:
        summarize = st.button("Summarize Document", use_container_width=True)

        with st.form("Analysis"):
            user_input = st.text_input('What questions do you have about this document?')
            submitted = st.form_submit_button("Submit", use_container_width=True)

        if summarize:
            with st.spinner("Summarizing document..."):
                try:
                    summary = doc_analysis(ss.pdf_text, user_input, user_choice='summary')
                    with st.container(height=1000):
                        st.markdown(summary)
                except Exception as e:
                    st.error(f"AI Error: {e}")

        if submitted and user_input:  
            with st.spinner("Analyzing document to answer question..."):
                try: 
                    analysis = doc_analysis(ss.pdf_text, user_input, user_choice='analyze')
                    with st.container(height=1000):
                        st.markdown(analysis)
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.info("Please upload a PDF file")

