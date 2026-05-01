import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------
# Configuration
# --------------------------------------------------
load_dotenv()
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS - ChatGPT/Claude Inspired UI
# --------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }

        .stApp {
            background: linear-gradient(#fbfdfb);
        }

        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .sub-title {
            text-align: center;
            color: #cbd5e1;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        .stChatMessage {
            border-radius: 18px;
            padding: 0.5rem;
            margin-bottom: 1rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            font-weight: 600;
            padding: 0.7rem 1rem;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.35);
        }

        .css-1d391kg, .css-1v0mbdj {
            background-color: rgba(15, 23, 42, 0.6);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = 0

# --------------------------------------------------
# LLM and Embeddings
# --------------------------------------------------
@st.cache_resource
def load_models():
    embedding_model = MistralAIEmbeddings()
    llm = ChatMistralAI(model="mistral-small-2506")
    return embedding_model, llm

embedding_model, llm = load_models()

# --------------------------------------------------
# Prompt Template
# --------------------------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

If document context is provided, prioritize it.
Use only the provided document context for document-specific questions.

If the answer is not present in the document context, clearly say:
'I could not find the answer in the document.'

If no document context is provided, answer normally like a general AI assistant.
"""
    ),
    (
        "human",
        """Document Context:
{context}

Question:
{question}
"""
    )
])

# --------------------------------------------------
# PDF Processing Function
# --------------------------------------------------
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks were created from the PDF.")

        persist_dir = "temp_chroma_db"

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        return vectorstore, retriever, len(chunks)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 Upload Your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploaded_file_key}"
    )

    if uploaded_file is not None:
        st.success(f"Selected: {uploaded_file.name}")

        if st.button("⚡ Process PDF"):
            with st.spinner("Processing PDF... Please wait."):
                try:
                    vectorstore, retriever, total_chunks = process_pdf(uploaded_file)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = retriever
                    st.session_state.pdf_processed = True
                    st.session_state.uploaded_file_key += 1

                    st.success(f"✅ PDF processed successfully! ({total_chunks} chunks created)")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.pdf_processed:
        st.markdown("---")
        st.success("📚 PDF is ready for questions!")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.markdown('<h1 class="main-title">PDF Chat Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload a PDF, ask questions from your document, or chat normally like ChatGPT.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
user_query = st.chat_input("Ask anything about your PDF or chat normally...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context = ""

                # Retrieve context if PDF is available
                if st.session_state.retriever is not None:
                    docs = st.session_state.retriever.invoke(user_query)
                    if docs:
                        context = "\n\n".join(doc.page_content for doc in docs)

                final_prompt = prompt_template.invoke({
                    "context": context if context else "No document context available.",
                    "question": user_query
                })

                response = llm.invoke(final_prompt)
                answer = response.content

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
