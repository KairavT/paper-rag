import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

USE_RERANK = True

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

title_to_file = {
    "Deep Reinforcement Learning from Human Preferences": "1706.03741",
    "Attention Is All You Need": "1706.03762",
    "Proximal Policy Optimization Algorithms": "1707.06347",
    "Training Language Models to Follow Instructions with Human Feedback": "2203.02155",
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model": "2305.18290",
    "Reinforcement Learning for LLM Post-Training: A Survey": "2407.16216",
    "Towards Revealing the Effectiveness of Small-Scale Fine-Tuning in R1-Style Reinforcement Learning": "2505.17988",
    "Reinforcement Learning for Large Model: A Survey": "2508.08189",
    "RL Is Neither a Panacea Nor a Mirage: Understanding Supervised vs. Reinforcement Learning Fine-Tuning for LLMs": "2508.16546",
    "School of Reward Hacks: Hacking Harmless Tasks Generalizes to Misaligned Behavior in LLMs": "2508.17511",
    "RL Fine-Tuning Heals OOD Forgetting in SFT": "2509.12235",
}

with open("eval_set.json", "r") as QAs:
    QA_list = json.load(QAs)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

chat = ChatOllama(model="llama3", temperature=0)

sources_used = 0
keys_in = 0
for QA in QA_list:

    Q = QA["question"]
    chunk = db.similarity_search(Q,k=20)
    
    if USE_RERANK:
        pairs = [[Q, c.page_content] for c in chunk]
        scores = reranker.predict(pairs)
        
        paired_chunks = zip(chunk, scores)
        paired_chunks = sorted(paired_chunks
                            , key=lambda pair: pair[1], reverse=True)
        paired_chunks = paired_chunks[:5]
        
        top5 = [pair[0] for pair in paired_chunks]
    else: 
        top5 = chunk[:5]

    ans_content = ''
    for c in top5:
        ans_content += f' {c.page_content}'

    prompt_qa =\
      f"Using the context {ans_content} and NOTHING ELSE,\
      answer the question \"{Q}\". If you don't \
        know the answer, state that you don't know instead \
            of guessing or trying to make something up."
    qa_answer = chat.invoke(prompt_qa)

    expected = title_to_file[QA["source"]]

    source_used = any(expected in c.metadata["source"] for c in top5)
    keyword_in = any(kw.lower() in qa_answer.content.lower() for kw in QA["keywords"])
    
    print(f"Question: {Q}\n"
          f"Expected: {QA['answer']}\n"
          f"Output: {qa_answer.content}\n"
          f"Expected Source Used: {source_used}\n"
          f"Contains Expected Keywords: {keyword_in}\n")
    
    

    if source_used: sources_used += 1
    if keyword_in: keys_in +=1


print(f'{sources_used}/{len(QA_list)} questions used the correct source,\
      {keys_in}/{len(QA_list)} questions contained an expected keyword')
