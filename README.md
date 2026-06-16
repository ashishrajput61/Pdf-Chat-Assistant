🤖 PDF Chat Assistant - RAG Application
<img width="1532" height="767" alt="Screenshot 2026-06-08 185844" src="https://github.com/user-attachments/assets/17119332-673b-4b00-b033-00d9d557c4c6" />

An intelligent Retrieval-Augmented Generation (RAG) powered chatbot that enables seamless conversation with PDF documents. Upload any PDF file and ask questions about its content in natural language. Powered by advanced LLMs and vector embeddings for accurate, context-aware responses.

Show Image


🎯 Project Overview

The PDF Chat Assistant is an intelligent document analysis tool that combines the power of:


Large Language Models (LLMs) - For natural language understanding and generation
Retrieval-Augmented Generation (RAG) - For accurate, context-based responses
Vector Embeddings - For semantic search and document understanding
Streamlit - For an intuitive, user-friendly interface


This application allows users to interact with PDF documents in a conversational manner, making document analysis faster and more intuitive than traditional reading.


✨ Key Features


📄 PDF Upload - Upload and process any PDF file instantly
💬 Natural Language Queries - Ask questions about your PDF in everyday language
🎯 Context-Aware Responses - Get accurate answers based on document content
🚀 Fast Processing - Quick extraction and indexing of PDF content
📊 Multi-Document Support - Process multiple documents in a session
🔍 Semantic Search - Understand the meaning, not just keywords
💾 Session Management - Maintain conversation history
🎨 Clean UI - Simple, intuitive Streamlit interface



🛠️ Technology Stack

Core Technologies:


Python 3.8+ - Programming language
Streamlit - Web application framework for rapid UI development
LangChain - Framework for LLM applications and RAG pipelines
OpenAI/LLM API - Language model for understanding and generation
FAISS/Vector DB - Vector database for semantic search
PyPDF - PDF processing and text extraction
Pandas - Data processing and manipulation


AI/ML Components:


Embedding Models - Convert text to semantic vectors
RAG Pipeline - Retrieval-Augmented Generation for accurate responses
LLM - Large Language Model for text generation
Vector Search - Semantic similarity matching



📋 How It Works

RAG Architecture:


PDF Upload → User uploads a PDF document
Text Extraction → Document text is extracted from PDF
Chunking → Text is split into manageable chunks
Embedding → Chunks are converted to vector embeddings
Vector Storage → Embeddings are stored in vector database
Query Processing → User question is embedded
Retrieval → Most relevant chunks are retrieved
Generation → LLM generates response based on retrieved context
Response → Answer is presented to user


User Query
    ↓
Query Embedding
    ↓
Vector Search (FAISS/Pinecone)
    ↓
Retrieved Documents
    ↓
LLM with Context
    ↓
Generated Response


🚀 Getting Started

Prerequisites:


Python 3.8 or higher
pip (Python package manager)
An API key from LLM provider (OpenAI, Anthropic, etc.)


Installation Steps:

1. Clone the Repository:

bashgit clone https://github.com/ashishrajput61/Pdf-Chat-Assistant.git
cd Pdf-Chat-Assistant

2. Create Virtual Environment (Optional but Recommended):

bashpython -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies:

bashpip install -r requirements.txt

4. Set Up Environment Variables:

Create a .env file in the root directory and add your API keys:

envOPENAI_API_KEY=your_openai_api_key_here
# or for other LLM providers:
# ANTHROPIC_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here

5. Run the Application:

bashstreamlit run app.py

The app will open in your default browser at http://localhost:8501


📱 Live Demo

Try the live application here:

🔗 PDF Chat Assistant - Streamlit App

No installation needed - just upload a PDF and start chatting!


💻 Usage Guide

Step-by-Step:


Open the Application

Visit the Streamlit app link above



Upload PDF Document

Click on "Upload PDF" button
Select a PDF file from your computer
Wait for processing to complete



Ask Questions

Type your question in the chat input
Ask follow-up questions naturally
Get context-aware responses



View Results

Read the generated response
Check referenced sections (if available)
Continue the conversation





Example Queries:


"What is the main topic of this document?"
"Summarize the key points"
"What does it say about [specific topic]?"
"Explain [concept] from the document"
"List all the requirements mentioned"



📁 Project Structure

Pdf-Chat-Assistant/
├── Rag Project/
│   ├── app.py                 # Main Streamlit application
│   ├── utils.py               # Utility functions
│   ├── config.py              # Configuration settings
│   └── vectordb.py            # Vector database operations
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file


📦 Required Dependencies

Key packages used in this project:

streamlit>=1.28.0
langchain>=0.1.0
openai>=1.0.0
pypdf>=3.0.0
python-dotenv>=1.0.0
faiss-cpu>=1.7.0  # or faiss-gpu for GPU acceleration
numpy>=1.24.0
pandas>=2.0.0

See requirements.txt for complete list.


🔑 API Keys Setup

For OpenAI (ChatGPT):


Visit OpenAI Platform
Sign up or log in
Navigate to API keys section
Create a new API key
Copy and paste in .env file


For Other LLM Providers:


Anthropic Claude: console.anthropic.com
Google Gemini: ai.google.dev
Cohere: dashboard.cohere.ai



⚙️ Configuration

You can customize the application by editing the configuration:

python# config.py
CHUNK_SIZE = 1000           # Size of text chunks
CHUNK_OVERLAP = 200         # Overlap between chunks
MAX_TOKENS = 500            # Max response length
TEMPERATURE = 0.7           # Response creativity (0-1)


🎓 Understanding RAG

What is Retrieval-Augmented Generation?

RAG combines two powerful approaches:


Retrieval - Finding relevant information from your documents
Generation - Using that information to generate accurate responses


Benefits:


✅ More accurate responses (grounded in your documents)
✅ Reduced hallucinations (no making up answers)
✅ Works with domain-specific documents
✅ Citeable sources (know where answers come from)
✅ Cost-effective (smaller context window needed)



🚀 Features in Detail

1. Intelligent Document Processing


Handles multi-page PDFs
Preserves document structure
Extracts metadata


2. Semantic Understanding


Understands context and meaning
Handles complex queries
Supports follow-up questions


3. Fast Responses


Vector-based search
Efficient embedding storage
Quick retrieval and generation


4. User-Friendly Interface


Streamlit-powered UI
Responsive design
Real-time feedback



🔒 Security & Privacy


Local Processing - Option to run locally
Environment Variables - API keys never hardcoded
Session-Based - Data not permanently stored
PDF Handling - No automatic cloud uploads (configurable)



📊 Performance Optimization

For Better Performance:

bash# Use GPU acceleration with FAISS
pip install faiss-gpu

# For faster embedding models
# Use smaller, optimized models
# Adjust chunk size based on document

# Cache embeddings
# Use vector database indices


🐛 Troubleshooting

Common Issues:

Issue: "API Key not found"


Solution: Check .env file exists and has correct key format


Issue: "PDF upload fails"


Solution: Ensure PDF is not corrupted, try with different PDF


Issue: "Slow responses"


Solution: Reduce chunk size, use GPU, check API quota


Issue: "Poor answer quality"


Solution: Check document relevance, adjust temperature, try different query phrasing



💡 Use Cases


📚 Research Documents - Analyze research papers and white papers
📖 Book Summaries - Get quick summaries of books
📋 Contract Analysis - Review and understand legal documents
📊 Report Analysis - Extract insights from business reports
🎓 Study Material - Interactive learning from textbooks
📰 News Articles - Detailed analysis of articles
🔬 Technical Documentation - Navigate complex documentation



🚀 Future Enhancements


 Support for multiple file formats (DOCX, TXT, images with OCR)
 Real-time conversation history
 Document comparison feature
 Multi-language support
 Advanced citation tracking
 Custom model fine-tuning
 Document annotation
 Team collaboration features
 Export conversation as PDF
 Voice input/output support



📊 Example Response

User Query: "What are the main conclusions of this document?"

System Process:


Convert query to embeddings
Search vector database for relevant sections
Retrieve top matching chunks
Send to LLM with context
Generate comprehensive response


Sample Response:


"Based on the document, the main conclusions are:


[Key finding 1 with specific evidence]
[Key finding 2 with specific evidence]
[Key finding 3 with specific evidence]


The document emphasizes... [additional insights]"




🤝 Contributing

We welcome contributions and suggestions!

How to Contribute:


Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Make your changes
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request


Areas for Improvement:


Additional LLM provider support
Performance optimizations
Enhanced UI/UX
Better error handling
Extended documentation



📝 License

This project is provided as-is for educational and commercial use.


👤 Author

Ashish Rajput


GitHub: @ashishrajput61
PDF Chat Assistant RAG Project



📞 Support & Questions


🌐 Live App: Streamlit Deployment
🐛 Issues: Open an issue on GitHub
💬 Discussions: Use GitHub Discussions for questions
📧 Email: Check GitHub profile for contact info



📚 Learning Resources

RAG & LLMs:


LangChain Documentation
OpenAI Documentation
Retrieval-Augmented Generation Paper


Vector Databases:


FAISS Guide
Pinecone Documentation
Weaviate Docs


Streamlit:


Streamlit Documentation
Streamlit Components



⭐ Show Your Support

If you find this project useful, please consider:


Starring the repository ⭐
Sharing with others in your network
Contributing improvements
Reporting issues and bugs



📈 Project Stats


Language: Python
Framework: Streamlit
Type: RAG Application
Status: Active Development
Last Updated: June 2026



🔗 Quick Links


🌐 Live Application
📦 GitHub Repository
📚 RAG Fundamentals
🔑 API Keys Setup Guide



🎉 Acknowledgments


Built with Streamlit for easy deployment
Powered by cutting-edge LLMs
Community feedback and support



Ready to chat with your PDFs? Start with the live demo now! 🚀


Last Updated: June 2026
An intelligent solution for document analysis and interaction.
