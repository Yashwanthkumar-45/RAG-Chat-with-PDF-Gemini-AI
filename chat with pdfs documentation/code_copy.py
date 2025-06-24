from PyPDF2 import PdfReader

def get_pdf_text(pdf_files):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text
