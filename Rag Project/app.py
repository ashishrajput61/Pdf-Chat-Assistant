import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
MISTRAL_API_KEY     = os.getenv("MISTRAL_API_KEY")
RELEVANCE_THRESHOLD = 0.40
MODEL_NAME          = "mistral-small-latest"
EMBED_MODEL         = "sentence-transformers/all-MiniLM-L6-v2"

# ── Prompts ───────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant.
Use ONLY the following document excerpts to answer the question.
If the answer is clearly present, answer thoroughly.
If you are unsure, say so honestly.

Context:
{context}

Question: {question}

Answer:""")

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable AI assistant. Answer the user's question based on your general knowledge. Be concise and helpful."),
    ("human", "{question}"),
])

# ── Text Extraction (with OCR fallback) ──────────────────────────────────────

def extract_text_pypdf(pdf_bytes: bytes) -> str:
    """Try normal text extraction first."""
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()


def extract_text_ocr(pdf_bytes: bytes, progress_bar) -> str:
    """
    OCR fallback for image-based / slide PDFs.
    Converts each page to image and runs Tesseract OCR.
    """
    images = convert_from_bytes(pdf_bytes, dpi=150)
    text = ""
    total = len(images)
    for i, img in enumerate(images):
        progress_bar.progress((i + 1) / total, text=f"OCR: page {i+1} of {total}…")
        page_text = pytesseract.image_to_string(img)
        if page_text.strip():
            text += page_text + "\n"
    return text.strip()


def extract_text(pdf_files, progress_bar) -> tuple[str, bool]:
    """
    Returns (extracted_text, ocr_was_used).
    Tries PyPDF2 first; falls back to OCR if no text found.
    """
    all_text = ""
    ocr_used = False

    for pdf in pdf_files:
        pdf_bytes = pdf.read()

        # Try normal extraction first
        text = extract_text_pypdf(pdf_bytes)

        if text:
            all_text += text + "\n"
        else:
            # No text layer found → OCR fallback
            ocr_used = True
            text = extract_text_ocr(pdf_bytes, progress_bar)
            all_text += text + "\n"

    return all_text.strip(), ocr_used


# ── Vector store & LLM ───────────────────────────────────────────────────────

def build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MODEL_NAME,
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def answer_query(question: str, vectorstore):
    llm = get_llm()

    if vectorstore:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=4)
        relevant_docs = [doc for doc, score in results_with_scores if score >= RELEVANCE_THRESHOLD]

        if relevant_docs:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | RAG_PROMPT
                | llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke(question)

            uncertain = any(p in answer.lower() for p in [
                "i don't know", "not mentioned", "not provided",
                "cannot find", "no information", "does not contain",
            ])
            if not uncertain:
                return answer, relevant_docs, True

    fallback_chain = FALLBACK_PROMPT | llm | StrOutputParser()
    answer = fallback_chain.invoke({"question": question})
    return answer, [], False


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []
if "uploaded_files_data" not in st.session_state:
    st.session_state.uploaded_files_data = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Chat Assistant")
st.caption("Answers come from your document when possible, or from AI knowledge otherwise.")

with st.sidebar:
    st.header("📁 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        st.session_state.uploaded_files_data = uploaded_files

    files_to_process = st.session_state.uploaded_files_data

    if files_to_process:
        st.markdown("**Selected files:**")
        for f in files_to_process:
            st.markdown(f"- 📄 {f.name}")

        if st.button("🔄 Process PDFs"):
            progress_bar = st.progress(0, text="Starting…")
            with st.spinner("Extracting & indexing…"):
                raw_text, ocr_used = extract_text(files_to_process, progress_bar)
                progress_bar.empty()

                if not raw_text:
                    st.error("Could not extract any text from the PDFs.")
                else:
                    st.session_state.vectorstore = build_vectorstore(raw_text)
                    st.session_state.pdf_names = [f.name for f in files_to_process]
                    if ocr_used:
                        st.info("ℹ️ OCR was used (image-based PDF detected).")
                    # Clear uploader
                    st.session_state.uploaded_files_data = None
                    st.session_state.uploader_key += 1
                    st.rerun()

    if st.session_state.pdf_names:
        st.divider()
        st.markdown("**Indexed files (ready to chat):**")
        for name in st.session_state.pdf_names:
            st.markdown(f"- ✅ {name}")

        if st.button("🗑️ Clear PDFs"):
            st.session_state.vectorstore = None
            st.session_state.pdf_names = []
            st.session_state.uploaded_files_data = None
            st.session_state.uploader_key += 1
            st.rerun()

    st.divider()
    st.markdown(
        "**Source legend**\n\n"
        "🟢 **From Document** — answer found in your PDF\n\n"
        "🔵 **General AI** — answer from model knowledge"
    )

# ── Chat ──────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("badge"):
            st.markdown(msg["badge"], unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📎 Source excerpts from PDF"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**Excerpt {i}:**\n> {chunk.page_content[:400]}…")

if user_query := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources, is_from_doc = answer_query(
                user_query, st.session_state.vectorstore
            )

        if is_from_doc:
            badge = '<span style="background:#22c55e;color:white;padding:2px 10px;border-radius:12px;font-size:0.78em;">🟢 Answer from Document</span>'
        else:
            badge = '<span style="background:#3b82f6;color:white;padding:2px 10px;border-radius:12px;font-size:0.78em;">🔵 General AI Response</span>'

        st.markdown(answer)
        st.markdown(badge, unsafe_allow_html=True)

        if sources:
            with st.expander("📎 Source excerpts from PDF"):
                for i, chunk in enumerate(sources, 1):
                    st.markdown(f"**Excerpt {i}:**\n> {chunk.page_content[:400]}…")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "badge": badge,
        "sources": sources,
    })
