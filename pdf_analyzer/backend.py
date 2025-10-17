import streamlit as st
import pymupdf
from streamlit_pdf_viewer import pdf_viewer
from openai import OpenAI
from dotenv import load_dotenv
import os

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

def doc_analysis(text, user_input, user_choice):
    client = OpenAI(api_key=open_ai_api_key)

    if user_choice.lower() == 'summary':
        user_prompt = 'Provide a comprehensive summary of this document'
    elif user_choice.lower() == 'complex_summary':
        user_prompt = 'Provide a detailed, complex analysis and summary of this document with technical insights and deep analysis'
    elif user_choice.lower() == 'simple_summary':
        user_prompt = 'Provide a very simple summary of this document using easy-to-understand language, as if explaining to a beginner'
    elif user_choice.lower() == 'analyze':
        user_prompt = user_input

    response = client.chat.completions.create(
        model='gpt-4.1',  
        messages=[
            {
                'role': 'system',
                'content': (
                    f"You are a professional yet friendly document analysis assistant. "
                    f"Analyze the document content and answer the user's question clearly and concisely. "
                    f"Document content: {text} " 
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

