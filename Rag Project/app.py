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


# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []
if "uploaded_files_data" not in st.session_state:
    st.session_state.uploaded_files_data = None
# uploader_key: incrementing this forces the file_uploader widget to fully reset
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Chat Assistant")
st.caption("Answers come from your document when possible, or from AI knowledge otherwise.")

with st.sidebar:
    st.header("📁 Upload PDFs")

    # Dynamic key — when incremented, Streamlit treats this as a brand new widget (fully empty)
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    # Save to session state as soon as user selects files
    if uploaded_files:
        st.session_state.uploaded_files_data = uploaded_files

    files_to_process = st.session_state.uploaded_files_data

    if files_to_process:
        st.markdown("**Selected files:**")
        for f in files_to_process:
            st.markdown(f"- 📄 {f.name}")

        if st.button("🔄 Process PDFs"):
            with st.spinner("Extracting & indexing…"):
                raw_text = extract_text(files_to_process)
                if not raw_text.strip():
                    st.error("No readable text found in the PDFs.")
                else:
                    st.session_state.vectorstore = build_vectorstore(raw_text)
                    st.session_state.pdf_names = [f.name for f in files_to_process]
                    # Clear stored files + bump key so uploader widget resets to empty
                    st.session_state.uploaded_files_data = None
                    st.session_state.uploader_key += 1
                    st.rerun()

    # Show indexed files (persists after rerun)
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
