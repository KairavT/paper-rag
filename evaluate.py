import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

with open("eval_set.json", "r") as QAs:
    QA_list = json.load(QAs)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

chat = ChatOllama(model="llama3")

for QA in QA_list:
    Q = QA["question"]
    chunk = db.similarity_search(Q,k=3)
    ans_content = ''
    for pg in chunk:
        ans_content += f' {pg.page_content}'
    prompt_qa =\
      f"Using the context {ans_content} and NOTHING ELSE,\
      answer the question \"{Q}\". If you don't \
        know the answer, state that you don't know instead \
            of guessing or trying to make something up."
    qa_answer = chat.invoke(prompt_qa)
    print(f"Question: {Q}\n"
          f"Expected: {QA['answer']}\n"
          f"Output: {qa_answer.content}\n")


