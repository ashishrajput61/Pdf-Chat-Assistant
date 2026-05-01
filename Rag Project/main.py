from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Embedding model
embedding_model = MistralAIEmbeddings()

# Load vector database
vectorstore = Chroma(
    persist_directory="chroma-DB",
    embedding_function=embedding_model
)

#Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5   # corrected spelling
    }
)

# print("Total vectors in DB:", vectorstore._collection.count())

# LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI Assistant.

Use only the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

print("RAG System created.")
print("Press 0 to Exit.")

while True:
    query = input("You: ")

    if query == "0":
        print("Exiting...")
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)
    print(f"\nRetrieved {len(docs)} documents.\n")
    if not docs:
        print("\nAI: No relevant documents found.")

    # Combine retrieved context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate response
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")