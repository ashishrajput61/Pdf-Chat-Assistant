import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RELEVANCE_THRESHOLD = 0.40   # cosine-similarity cutoff (0–1); tune as needed
MODEL_NAME       = "llama3-8b-8192"   # change to any Groq model you prefer
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"

# ── Prompt for RAG (document-grounded) answers ──────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the following document excerpts to answer the question.
If the answer is clearly present in the context, answer it thoroughly.
If you are not sure, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
)

# ── Prompt for general AI fallback ──────────────────────────────────────────
FALLBACK_SYSTEM = (
    "You are a knowledgeable AI assistant. "
    "Answer the user's question based on your general knowledge. "
    "Be concise and helpful."
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_text_from_pdfs(pdf_files) -> str:
    """Extract raw text from uploaded PDF file objects."""
    full_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def get_llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3,
    )


def answer_from_document(query: str, vectorstore: FAISS):
    """
    Try to answer from the PDF.
    Returns (answer, source_chunks, is_from_doc).
    """
    # Similarity search with score (lower L2 = more similar for FAISS)
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=4)

    if not results_with_scores:
        return None, [], False

    # Filter by threshold
    relevant_docs = [doc for doc, score in results_with_scores if score >= RELEVANCE_THRESHOLD]

    if not relevant_docs:
        return None, [], False

    # Build RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )
    result = chain.invoke({"query": query})
    answer = result.get("result", "").strip()

    # If the LLM says it doesn't know despite docs existing, fallback
    uncertain_phrases = [
        "i don't know", "not mentioned", "not provided",
        "cannot find", "no information", "does not contain",
    ]
    is_uncertain = any(p in answer.lower() for p in uncertain_phrases)
    if is_uncertain:
        return answer, result.get("source_documents", []), False

    return answer, result.get("source_documents", []), True


def answer_from_ai(query: str) -> str:
    """General AI answer when no document context is relevant."""
    llm = get_llm()
    messages = [
        ("system", FALLBACK_SYSTEM),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")

st.title("📄 PDF Chat Assistant")
st.caption("Ask anything — answers come from your document when possible, or from AI knowledge otherwise.")

# Sidebar: PDF upload
with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🔄 Process PDFs"):
            with st.spinner("Extracting & indexing…"):
                raw_text = extract_text_from_pdfs(uploaded_files)
                if not raw_text.strip():
                    st.error("No readable text found in the PDFs.")
                else:
                    st.session_state.vectorstore = build_vectorstore(raw_text)
                    st.session_state.pdf_names = [f.name for f in uploaded_files]
                    st.success(f"✅ Indexed {len(uploaded_files)} file(s)!")

    if "pdf_names" in st.session_state:
        st.markdown("**Loaded files:**")
        for name in st.session_state.pdf_names:
            st.markdown(f"- 📄 {name}")

    st.divider()
    st.markdown(
        "**Source legend**\n\n"
        "🟢 **From Document** — answer found in your PDF\n\n"
        "🔵 **General AI** — answer from model knowledge"
    )

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source_badge"):
            st.markdown(msg["source_badge"], unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📎 Source excerpts from PDF"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**Excerpt {i}:**\n> {chunk.page_content[:400]}…")

# Chat input
user_query = st.chat_input("Ask a question…")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            vectorstore = st.session_state.get("vectorstore", None)

            if vectorstore:
                answer, source_docs, is_from_doc = answer_from_document(
                    user_query, vectorstore
                )
                if not is_from_doc:
                    # Supplement with general AI
                    ai_answer = answer_from_ai(user_query)
                    final_answer = ai_answer
                    badge = (
                        '<span style="background:#3b82f6;color:white;'
                        'padding:2px 10px;border-radius:12px;font-size:0.78em;">'
                        "🔵 General AI Response</span>"
                    )
                    source_docs = []
                else:
                    final_answer = answer
                    badge = (
                        '<span style="background:#22c55e;color:white;'
                        'padding:2px 10px;border-radius:12px;font-size:0.78em;">'
                        "🟢 Answer from Document</span>"
                    )
            else:
                # No PDF uploaded — pure AI
                final_answer = answer_from_ai(user_query)
                badge = (
                    '<span style="background:#3b82f6;color:white;'
                    'padding:2px 10px;border-radius:12px;font-size:0.78em;">'
                    "🔵 General AI Response</span>"
                )
                source_docs = []

        st.markdown(final_answer)
        st.markdown(badge, unsafe_allow_html=True)

        if source_docs:
            with st.expander("📎 Source excerpts from PDF"):
                for i, chunk in enumerate(source_docs, 1):
                    st.markdown(f"**Excerpt {i}:**\n> {chunk.page_content[:400]}…")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "source_badge": badge,
        "sources": source_docs,
    })
