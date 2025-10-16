# PDF Analyzer

A Streamlit web application that allows users to upload PDF files and analyze them using OpenAI's GPT models.

## Features

- Upload and view PDF files
- Extract text from PDFs
- AI-powered document summarization
- Ask questions about document content
- Real-time progress tracking

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