import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
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
MODEL_NAME          = "mistral-small-latest"   # or "mistral-large-latest"
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(pdf_files) -> str:
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


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
    """
    Returns (answer, source_docs, is_from_doc).
    Tries document first; falls back to general AI.
    """
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

    # Fallback: general AI
    fallback_chain = FALLBACK_PROMPT | llm | StrOutputParser()
    answer = fallback_chain.invoke({"question": question})
    return answer, [], False


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Chat Assistant")
st.caption("Answers come from your document when possible, or from AI knowledge otherwise.")

with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("🔄 Process PDFs"):
        with st.spinner("Extracting & indexing…"):
            raw_text = extract_text(uploaded_files)
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

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
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
                user_query, st.session_state.get("vectorstore")
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
