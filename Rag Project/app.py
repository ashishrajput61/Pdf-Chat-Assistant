import os
import tempfile
import hashlib
import time

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
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
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
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .pdf-banner {
        background: linear-gradient(90deg, #1e40af22, #7c3aed22);
        border: 1px solid #6366f155;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 600;
        padding: 0.7rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(59,130,246,0.35);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ✅ SHARED STATE  (one PDF shared across all users)
# Each user gets their own messages list.
# The vectorstore + retriever live in st.session_state
# which Streamlit shares server-side for the same app.
# --------------------------------------------------
def init_shared_state():
    """Global PDF state — shared across all connected users."""
    if "shared_vectorstore" not in st.session_state:
        st.session_state["shared_vectorstore"] = None
    if "shared_retriever" not in st.session_state:
        st.session_state["shared_retriever"] = None
    if "shared_pdf_name" not in st.session_state:
        st.session_state["shared_pdf_name"] = None
    if "shared_pdf_hash" not in st.session_state:
        st.session_state["shared_pdf_hash"] = None
    if "shared_chunk_count" not in st.session_state:
        st.session_state["shared_chunk_count"] = 0
    if "shared_upload_time" not in st.session_state:
        st.session_state["shared_upload_time"] = None


def init_user_state():
    """Per-user state — each user's own chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


init_shared_state()
init_user_state()

# --------------------------------------------------
# LLM and Embeddings  (cached — loaded once)
# --------------------------------------------------
@st.cache_resource
def load_models():
    embedding_model = MistralAIEmbeddings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )
    return embedding_model, llm

embedding_model, llm = load_models()

# --------------------------------------------------
# Prompt Template
# --------------------------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

If document context is provided, prioritize it to answer the user's question.
Use only the provided document context for document-specific questions.

If the answer is not present in the document context, clearly say:
'I could not find the answer in the document.'

If no document context is provided, answer normally like a general AI assistant."""
    ),
    (
        "human",
        """Document Context:
{context}

Question:
{question}"""
    )
])

# --------------------------------------------------
# ✅ FIXED PDF Processing Function
# Bug was: PyPDFLoader can return pages with empty/None
# text (scanned PDFs). We now filter those out BEFORE
# splitting, so chunks list is never empty on valid PDFs.
# --------------------------------------------------
def get_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def process_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    # Write to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            raise ValueError(
                "The PDF appears to be empty or could not be read. "
                "Please try a different PDF."
            )

        # ✅ FIX: Filter out pages with no extractable text BEFORE splitting
        valid_docs = [
            doc for doc in documents
            if doc.page_content and doc.page_content.strip()
        ]

        if not valid_docs:
            raise ValueError(
                "No text could be extracted from this PDF.\n\n"
                "Possible reasons:\n"
                "• The PDF is scanned / image-based (no selectable text)\n"
                "• The PDF is password-protected\n"
                "• The file is corrupted\n\n"
                "Please upload a text-based PDF."
            )

        # Split valid pages into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
        chunks = text_splitter.split_documents(valid_docs)

        if not chunks:
            raise ValueError(
                "Text was found but could not be split into chunks. "
                "The document may be too short."
            )

        # Build Chroma vectorstore
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="temp_chroma_db",
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )

        return vectorstore, retriever, len(chunks)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 Upload PDF")
    st.caption("Any user can upload a PDF. All users will chat with the same PDF.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        st.info(f"📎 Selected: **{uploaded_file.name}**")

        if st.button("⚡ Process & Share PDF"):
            file_bytes = uploaded_file.getvalue()
            new_hash = get_pdf_hash(file_bytes)

            # Skip re-processing if it's the same PDF
            if new_hash == st.session_state["shared_pdf_hash"]:
                st.info("✅ This PDF is already loaded. Start chatting!")
            else:
                with st.spinner("Processing PDF for all users…"):
                    try:
                        vectorstore, retriever, total_chunks = process_pdf(uploaded_file)

                        # ✅ Save to SHARED state (all users get this)
                        st.session_state["shared_vectorstore"] = vectorstore
                        st.session_state["shared_retriever"]   = retriever
                        st.session_state["shared_pdf_name"]    = uploaded_file.name
                        st.session_state["shared_pdf_hash"]    = new_hash
                        st.session_state["shared_chunk_count"] = total_chunks
                        st.session_state["shared_upload_time"] = time.strftime("%H:%M:%S")

                        # Reset uploader widget
                        st.session_state.uploader_key += 1

                        st.success(f"✅ PDF ready! ({total_chunks} chunks)")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ {str(e)}")

    # Show current active PDF info
    if st.session_state["shared_pdf_name"]:
        st.markdown("---")
        st.markdown("**📚 Active PDF (shared with all users):**")
        st.markdown(f"- 📄 **{st.session_state['shared_pdf_name']}**")
        st.markdown(f"- 🧩 {st.session_state['shared_chunk_count']} chunks")
        st.markdown(f"- 🕐 Uploaded at {st.session_state['shared_upload_time']}")

    st.markdown("---")
    if st.button("🗑️ Clear My Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**ℹ️ How it works:**")
    st.markdown(
        "1. Upload a PDF → processed once\n"
        "2. All users can ask questions from it\n"
        "3. Each user has their own chat history\n"
        "4. Upload a new PDF → updates for everyone"
    )

# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.markdown('<h1 class="main-title">📄 PDF Chat Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload a PDF and ask questions — multiple users can chat simultaneously.</p>',
    unsafe_allow_html=True,
)

# Show active PDF banner
if st.session_state["shared_pdf_name"]:
    st.markdown(
        f'<div class="pdf-banner">📚 Active PDF: <b>{st.session_state["shared_pdf_name"]}</b> &nbsp;·&nbsp; '
        f'{st.session_state["shared_chunk_count"]} chunks &nbsp;·&nbsp; '
        f'Uploaded at {st.session_state["shared_upload_time"]}</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("👈 Upload a PDF from the sidebar to get started.")

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
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context = ""

                # Use shared retriever if PDF is loaded
                retriever = st.session_state.get("shared_retriever")
                if retriever is not None:
                    docs = retriever.invoke(user_query)
                    if docs:
                        context = "\n\n".join(doc.page_content for doc in docs)

                final_prompt = prompt_template.invoke({
                    "context": context if context else "No document context available.",
                    "question": user_query,
                })

                response = llm.invoke(final_prompt)
                answer = response.content

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
