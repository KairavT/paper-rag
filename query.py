from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

chunks = db.similarity_search("How is multi-head attention used in transformers?",
                                k=3)

for page in chunks:
    print(page.page_content)
