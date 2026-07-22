from langchain_ollama import ChatOllama

chat = ChatOllama(model ="llama3")
test_msg  = chat.invoke("How many letters are in this sentence?")
print(test_msg.content)
