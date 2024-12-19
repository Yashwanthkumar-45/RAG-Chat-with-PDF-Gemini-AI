RAG Chat with PDF Using Gemini AI
A RAG (Retrieve-Answer-Generate) application designed to enable users to chat with PDF documents using the power of Gemini AI. This project leverages advanced machine learning and natural language processing (NLP) techniques to retrieve relevant information from PDF files and answer user queries interactively.

Features
• Upload and process PDF files.
• Retrieve answers to user questions based on PDF content.
• Interactive chatbot interface for real-time querying.
• RAG pipeline implementation for improved accuracy and relevance.
• Streamlit-powered web interface for user-friendly operation.

Technologies Used
• Python: Core programming language.
• Streamlit: For building the web-based user interface.
• LangChain: For managing the RAG pipeline and embeddings.
• FAISS: For vector similarity search.
• OpenAI API: For language generation and answering user queries.
• Pickle: For serializing and deserializing index files.

Setup Instructions
Prerequisites
• Python 3.8 or later
• Pip package manager
• OpenAI API key
Installation Steps
1. Clone this repository:
bash
Copy code
git clone https://github.com/Yashwanthkumar-45/RAG-Chat-with-PDF-Gemini-AI.git  
2. Navigate to the project directory:
bash
Copy code
cd RAG-Chat-with-PDF-Gemini-AI  
3. Create and activate a virtual environment:
bash
Copy code
python -m venv venv  
source venv/bin/activate        # On Linux/Mac  
venv\Scripts\activate          # On Windows  
4. Install dependencies:
bash
Copy code
pip install -r requirements.txt  
5. Add your OpenAI API key in the code or as environment variables.

Usage Instructions
1. Start the Streamlit app:
bash
Copy code
streamlit run app.py  
2. Open the app in your browser at http://localhost:8501.
3. Interact with the chatbot:
o Upload a PDF file.
o Ask questions and get answers based on the PDF content.

File Structure
plaintext
Copy code
RAG-Chat-with-PDF-Gemini-AI/  
├── app.py             # Main application script  
├── requirements.txt   # Dependencies for the project  
├── faiss_index/       # Directory to store FAISS index  
├── static/            # Static files (if any)  
├── .gitignore         # Git ignore file  
└── README.md          # Documentation  

Known Issues
• ValueError: If a deserialization error occurs, set allow_dangerous_deserialization=True (trusted data only).
• Ensure FAISS is correctly installed and configured for your platform.

Future Enhancements
• Add support for multiple file uploads.
• Improve chatbot conversational capabilities.
• Extend to handle other file formats like Word or Excel.

Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License.

Acknowledgments
• OpenAI for their powerful APIs.
• Streamlit for enabling rapid UI development.
• LangChain and FAISS for robust data handling and vector similarity search.

