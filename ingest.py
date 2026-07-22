from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFDirectoryLoader('papers')
loaded = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

post_split = splitter.split_documents(loaded)
print(post_split[0].page_content)

load_size = len(loaded)
split_size = len(post_split)
print(load_size, split_size)