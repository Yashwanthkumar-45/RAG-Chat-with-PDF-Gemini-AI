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

# Tesseract path setup for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    """Extracts text from PDFs using PyPDF2 and OCR if needed."""
    pdf_text = ''
    for pdf_doc in pdf_docs:
        try:
            # Read the file as bytes
            pdf_bytes = pdf_doc.read()

            # Try extracting text using PyPDF2
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            text_extracted = False
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
                    text_extracted = True

            if not text_extracted:  # If no text extracted, use OCR
                st.warning(f"No text extracted from {pdf_doc.name}. Attempting OCR...")

                # Save PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
                    temp_pdf_file.write(pdf_bytes)
                    temp_pdf_path = temp_pdf_file.name

                # Convert PDF pages to images and perform OCR
                images = convert_from_path(temp_pdf_path)
                for image in images:
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        pdf_text += ocr_text
                    else:
                        st.warning("OCR did not detect any text.")

                # Remove temporary file
                os.remove(temp_pdf_path)

        except Exception as e:
            st.error(f"Error processing {pdf_doc.name}: {e}")
    return pdf_text

def get_text_chunks(text):
    """Splits text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks:
        st.error("Text chunks are empty. Ensure the uploaded PDF contains readable text.")
        return None

    try:
        # Use the correct model name
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        sample_embedding = embeddings.embed_query("test")  # Test the embedding

        if not sample_embedding:
            raise ValueError("Failed to generate embeddings. Check the API key or model configuration.")

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error in generating vector store: {e}")
        return None


def get_conversational_chain():
    """Returns a conversational chain for QA."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, respond interactively:
    - "Hmm, I couldn't find an answer in the documents. Could you rephrase?"
    - "Sorry, no relevant details found. Please ask in a different way."
    - "It seems like the answer isn't available in the provided context. Can you try another question?"
    
    Context: {context}?
    Question: {question}

    Answer:"""

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Handles user questions and returns the response."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        fallback_responses = [
            "Hmm, I couldn't find an answer in the documents. Could you rephrase?",
            "Sorry, no relevant details found. Please ask differently.",
            "It seems like the answer isn't available. Try another question."
        ]

        if response:
            output_text = response.get("output_text", "").strip()
            if output_text:
                st.write("Reply: ", output_text)
            else:
                st.warning(random.choice(fallback_responses))
        else:
            st.warning(random.choice(fallback_responses))

    except KeyError:
        st.warning("Response format is unexpected. Please try rephrasing your question.")
    except Exception as e:
        st.error(f"An error occurred while processing the query: {e}")

def main():
    """Main application logic."""
    st.set_page_config(page_title="Chat with PDFs")
    st.header("Chat with PDFs Using Gemini AI")

    user_question = st.text_input("Ask any question related to the uploaded PDFs")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text found. Please upload a valid PDF.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    if vector_store:
                        st.success("Processing completed successfully.")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
