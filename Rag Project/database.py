from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Load PDF
loader = PyPDFLoader(r"C:\Users\rajpu\OneDrive\Desktop\Gen Ai\Rag Project\Book.pdf")
documents = loader.load()

print(f"Loaded documents: {len(documents)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

if not chunks:
    raise ValueError("No text chunks were created. Check your PDF.")

# Create embeddings
embedding_model = MistralAIEmbeddings()

# Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-DB"
)

print("Chroma DB created successfully!")
print(f"Stored {vectorstore._collection.count()} chunks.")