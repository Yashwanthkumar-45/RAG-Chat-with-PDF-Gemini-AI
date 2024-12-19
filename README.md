**RAG Chat with PDF Using Gemini AI** <br></br>
A RAG (Retrieval Augmented Generation) application designed to enable users to chat with PDF documents using the power of Gemini AI. This project leverages advanced machine learning and natural language processing (NLP) techniques to retrieve relevant information from PDF files and answer user queries interactively.

**Features**<br></br>
• Upload and process PDF files.<br></br>
• Retrieve answers to user questions based on PDF content.<br></br>
• Interactive chatbot interface for real-time querying.<br></br>
• RAG pipeline implementation for improved accuracy and relevance.<br></br>
• Streamlit-powered web interface for user-friendly operation.<br></br>

**Technologies Used**<br></br>
• Python: Core programming language.<br></br>
• Streamlit: For building the web-based user interface.<br></br>
• LangChain: For managing the RAG pipeline and embeddings.<br></br>
• FAISS: For vector similarity search.<br></br>
• OpenAI API: For language generation and answering user queries.<br></br>
• Pickle: For serializing and deserializing index files.<br></br>

**Setup Instructions**<br></br>
Prerequisites<br></br>
• Python 3.8 or later<br></br>
• Pip package manager<br></br>
• OpenAI API key<br></br>
Installation Steps<br></br>
1. Clone this repository:<br></br>
git clone https://github.com/Yashwanthkumar-45/RAG-Chat-with-PDF-Gemini-AI.git  
2. Navigate to the project directory:<br></br>
cd RAG-Chat-with-PDF-Gemini-AI  
3. Create and activate a virtual environment:<br></br>
python -m venv venv  
source venv/bin/activate        # On Linux/Mac  
venv\Scripts\activate          # On Windows  
4. Install dependencies:<br></br>
pip install -r requirements.txt  
5. Add your OpenAI API key in the code or as environment variables.<br></br>

**Usage Instructions**<br></br>
1. Start the Streamlit app:<br></br>
streamlit run app.py  
2. Open the app in your browser at http://localhost:8501.
3. Interact with the chatbot:<br></br>
o Upload a PDF file.<br></br>
o Ask questions and get answers based on the PDF content.<br></br>

**File Structure**<br></br>
RAG-Chat-with-PDF-Gemini-AI/  
├── app.py             # Main application script  
├── requirements.txt   # Dependencies for the project  
├── faiss_index/       # Directory to store FAISS index  
├── static/            # Static files (if any)  
├── .gitignore         # Git ignore file  
└── README.md          # Documentation  

**Known Issues**<br></br>
• ValueError: If a deserialization error occurs, set allow_dangerous_deserialization=True (trusted data only).<br></br>
• Ensure FAISS is correctly installed and configured for your platform.<br></br>

**Contributing**<br></br>
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

**License**<br></br>
This project is licensed under the MIT License.

**Acknowledgments**<br></br>
• OpenAI for their powerful APIs.
• Streamlit for enabling rapid UI development.
• LangChain and FAISS for robust data handling and vector similarity search.

