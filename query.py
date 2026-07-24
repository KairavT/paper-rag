from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"
)
question = "How is multi-head attention used in transformers?"
chunks = db.similarity_search(question,
                                k=3)


content = ''
for page in chunks:
    content += f' {page.page_content}'

prompt =\
      f"Using the context {content} and NOTHING ELSE,\
      answer the question \"{question}\"."

chat = ChatOllama(model="llama3")
context_answer = chat.invoke(prompt)
print(context_answer.content)
