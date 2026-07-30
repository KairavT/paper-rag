# paper-rag

A retrieval-augmented generation (RAG) system that answers questions about a
collection of machine learning papers, running entirely locally with Ollama. Ask
a question, and it retrieves the most relevant passages from the papers and uses
a local language model to answer from them.

## What it does

The corpus is a set of 11 arXiv papers on reinforcement learning, RLHF, and
transformers. Given a question like "what problem does PPO solve?", the system
finds the passages in those papers most relevant to the question and generates an
answer grounded in them, rather than relying on the language model's own memory.
Everything runs on your machine with no paid API.

## How it works

The pipeline has two stages.

Ingestion (run once) reads every PDF, splits each into overlapping chunks,
converts each chunk into an embedding vector with a local embedding model, and
stores the vectors in a local Chroma database.

Querying (run per question) embeds the question, retrieves the most similar
chunks from the store, and passes those chunks plus the question to a local
language model with an instruction to answer only from the provided context.

The models are served by Ollama: llama3 for generation and nomic-embed-text for
embeddings.

## Evaluation

The project includes an evaluation harness that measures quality rather than
guessing at it. It runs a set of 20 questions whose answers are known and checked
against the actual paper text, and reports two numbers: how often the correct
source paper was retrieved, and how often the generated answer contained the
expected facts. Evaluation runs deterministically so results are reproducible.

This harness is what makes the system tunable. Retrieval settings were compared
against it rather than by eye, which surfaced a few things worth knowing:

- Chunk size matters. Chunks that are too small split a single fact across
  pieces and hurt retrieval; larger chunks that keep concepts intact worked
  better.
- Retrieving more candidate chunks did not keep improving results past a point.
- Cross-encoder reranking, a common add-on, gave no clear benefit on this small
  corpus of distinct papers, while adding noticeable latency.

## Known limitations

- The system is good at facts that are stated prominently or repeated, and weak
  at single-mention specifics buried deep in a paper (an exact hyperparameter, a
  one-line optimizer choice). Those chunks are harder to surface.
- On definitional questions, a survey paper that discusses a concept explicitly
  can out-rank the original paper that introduced it, so the answer sometimes
  comes from the wrong source even when it is correct.
- Answer scoring is keyword-based, which is transparent but can occasionally miss
  a correctly-phrased answer or accept a wrongly-sourced one.
- Local models are weaker than hosted ones, so answer quality reflects the model,
  not just the retrieval.

Metadata filtering (for example, excluding survey papers on definitional queries)
is a natural next step to address the source-ranking issue, and is left as future
work.

## Setup

```
git clone https://github.com/KairavT/paper-rag.git
cd paper-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install Ollama and pull the models:

```
ollama pull llama3
ollama pull nomic-embed-text
```

## Usage

Build the vector store once from the PDFs in the papers folder:

```
python ingest.py
```

Ask a question:

```
python query.py
```

Run the evaluation:

```
python evaluate.py
```

## Tech stack

Python, LangChain, Ollama (llama3, nomic-embed-text), Chroma, sentence-transformers.
