

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "chroma-DB"


print("Loading models...")
embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
)

total = vectorstore._collection.count()
if total == 0:
    print(
        " Chroma DB is empty. Run database.py first:\n"
        "   python database.py --pdf your_file.pdf"
    )
    exit(1)

print(f" Loaded Chroma DB with {total} chunks.")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
)

llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI Assistant.

Use only the provided context to answer the question.
If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}""",
    ),
])

print("\n RAG Chat ready. Type your question or press 0 to exit.\n")

while True:
    query = input("You: ").strip()

    if query == "0":
        print("Exiting...")
        break

    if not query:
        continue

    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: No relevant documents found.\n")
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
