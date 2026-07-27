from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

loader = PyPDFDirectoryLoader('papers')
loaded = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

post_split = splitter.split_documents(loaded)
#print(post_split[0].page_content)

load_size = len(loaded)
split_size = len(post_split)
#print(load_size, split_size)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma.from_documents(
    documents = post_split,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print(db._collection.count())
