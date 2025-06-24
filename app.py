import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
import io
import tempfile
import platform
from typing import List

# Configure Tesseract path based on OS
def configure_tesseract():
    system = platform.system()
    if system == "Windows":
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    elif system == "Linux":
        default_path = r"/usr/bin/tesseract"
    elif system == "Darwin":  # macOS
        default_path = r"/usr/local/bin/tesseract"
    else:
        default_path = "tesseract"
    
    # Allow override with environment variable
    return os.getenv('TESSERACT_PATH', default_path)

pytesseract.pytesseract.tesseract_cmd = configure_tesseract()

# Load environment variables
load_dotenv()

# Configure GenAI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

#@st.cache_data(show_spinner=False)
def is_pdf_searchable(pdf_bytes: bytes) -> bool:
    """Check if PDF contains text layers."""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            resources = page.get("/Resources", {})
            if "/Font" in resources:
                return True
        return False
    except Exception:
        return False

#@st.cache_data(show_spinner="Processing PDF...")
def get_pdf_text(pdf_docs: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Extracts text from PDFs using PyPDF2 and OCR if needed."""
    pdf_text = ''
    
    for pdf_doc in pdf_docs:
        try:
            with st.spinner(f"Processing {pdf_doc.name}..."):
                st.toast(f"Processing {pdf_doc.name}", icon="📄")
                pdf_bytes = pdf_doc.read()

                # First try to extract text directly
                if is_pdf_searchable(pdf_bytes):
                    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text
                            st.toast(f"Page {i+1} extracted", icon="✅")
                        else:
                            st.toast(f"Page {i+1} has no text layer", icon="⚠️")
                
                # If no text found, try OCR
                if not pdf_text.strip():
                    st.warning(f"No text extracted from {pdf_doc.name}. Attempting OCR...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
                        temp_pdf_file.write(pdf_bytes)
                        temp_pdf_path = temp_pdf_file.name

                    try:
                        images = convert_from_path(temp_pdf_path)
                        progress_bar = st.progress(0)
                        for i, image in enumerate(images):
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text.strip():
                                pdf_text += ocr_text
                                st.toast(f"Page {i+1} OCR completed", icon="🔍")
                            else:
                                st.warning(f"OCR found no text on page {i+1}")
                            progress_bar.progress((i + 1) / len(images))
                        progress_bar.empty()
                    finally:
                        os.remove(temp_pdf_path)

        except Exception as e:
            st.error(f"Error processing {pdf_doc.name}: {str(e)}")
    
    if not pdf_text.strip():
        st.error("No readable text found in any of the uploaded documents.")
    return pdf_text

@st.cache_data(show_spinner="Chunking text...")
def get_text_chunks(text: str) -> List[str]:
    """Splits text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache_resource(show_spinner=False)  # Removed UI elements from this function
def get_vector_store(text_chunks: List[str]):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks:
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error in generating vector store: {str(e)}")
        return None

def get_conversational_chain():
    """Create QA chain with properly formatted safety settings."""
    prompt_template = """
    Answer the question from context. If unsure, say:
    "I couldn't find a definitive answer. Try rephrasing."
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    # Correct numeric format for safety settings
    safety_settings = {
        1: 0,  # HARM_CATEGORY_HARASSMENT: BLOCK_NONE
        2: 0,  # HARM_CATEGORY_HATE_SPEECH: BLOCK_NONE
        3: 0,  # HARM_CATEGORY_SEXUALLY_EXPLICIT: BLOCK_NONE
        4: 0   # HARM_CATEGORY_DANGEROUS_CONTENT: BLOCK_NONE
    }
    
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            safety_settings=safety_settings,
            max_output_tokens=2048
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return load_qa_chain(
            model,
            chain_type="stuff",
            prompt=prompt
        )
        
    except Exception as e:
        st.error(f"Failed to create conversation chain: {str(e)}")
        raise RuntimeError("Could not initialize model") from e

def handle_user_query(user_question: str):
    """Handles user questions and displays responses."""
    try:
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process documents first.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        docs = new_db.similarity_search(user_question, k=5)
        chain = get_conversational_chain()
        
        with st.spinner("Analyzing documents..."):
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

        if response and "output_text" in response:
            output_text = response["output_text"].strip()
            if output_text:
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                st.chat_message("assistant").write(output_text)
                return
        
        # Fallback responses
        fallback_responses = [
            "I couldn't find a definitive answer in the documents. Could you rephrase or provide more context?",
            "The documents don't appear to contain a clear answer to this. Would you like to ask about something else?",
            "I'm not finding relevant information for this question in the uploaded files."
        ]
        fallback = random.choice(fallback_responses)
        st.session_state.messages.append({"role": "assistant", "content": fallback})
        st.chat_message("assistant").write(fallback)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def display_chat():
    """Displays chat messages from session state."""
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

def process_uploaded_files(pdf_docs):
    """Handles the file processing pipeline with proper UI separation."""
    with st.status("Processing documents...", expanded=True) as status:
        st.write("Extracting text from PDFs...")
        raw_text = get_pdf_text(pdf_docs)
        
        if not raw_text.strip():
            status.update(label="Processing failed - no text found", state="error")
            return False

        st.write("Splitting text into chunks...")
        text_chunks = get_text_chunks(raw_text)
        
        st.write("Creating vector store...")
        try:
            vector_store = get_vector_store(text_chunks)
            if vector_store:
                status.update(label="Processing complete!", state="complete")
                st.session_state.processed = True
                return True
            else:
                status.update(label="Processing failed", state="error")
                return False
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            return False

def main():
    """Main application logic."""
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📄",
        layout="centered"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    st.title("📄 Chat with PDFs")
    st.caption("Upload PDF documents and ask questions about their content")
    
    # Display chat messages
    display_chat()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.get("processed", False):
            st.warning("Please upload and process documents first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            handle_user_query(prompt)

    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Processing")
        st.markdown("""
        **Instructions:**
        1. Upload PDF files
        2. Click 'Process Documents'
        3. Ask questions in the chat
        """)
        
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type="pdf",
            help="Upload one or more PDF documents to analyze"
        )
        
        if st.button("Process Documents", type="primary"):
            if pdf_docs:
                if process_uploaded_files(pdf_docs):
                    st.success("Documents ready for questioning!")
            else:
                st.warning("Please upload at least one PDF file.")
        
        st.divider()
        st.markdown("""
        **Note:** For image-based PDFs, the app will automatically use OCR.
        Processing may take longer for large documents.
        """)

if __name__ == "__main__":
    main()