
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


def build_vectorstore(pdf_path: str, persist_dir: str = "chroma-DB"):
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loading PDF: {path.name}")
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")
    valid_docs = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    if not valid_docs:
        raise ValueError(
            "No text could be extracted from this PDF.\n"
            "It may be scanned/image-based. Use a text-based PDF."
        )

    print(f"  Valid pages with text: {len(valid_docs)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(valid_docs)
    print(f"  Total chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks created. Document may be too short.")

    print("Creating embeddings and saving to Chroma DB...")
    embedding_model = MistralAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )

    print(f" Chroma DB saved to '{persist_dir}'")
    print(f"   Stored {vectorstore._collection.count()} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma vectorstore from a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--dir", default="chroma-DB", help="Chroma persist directory")
    args = parser.parse_args()

    build_vectorstore(args.pdf, args.dir)
