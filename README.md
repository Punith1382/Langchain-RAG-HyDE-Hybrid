# Langchain-RAG-HyDE-Hybrid

# LangChain RAG Toolkit — Standard RAG, HyDE, and Hybrid Search (BM25 + Vector)

A progressive, notebook-based walkthrough of building Retrieval-Augmented Generation
(RAG) systems with **LangChain**, moving from basic LLM calls through three complete
retrieval strategies: standard vector RAG, **HyDE**, and **Hybrid Search** (BM25 +
vector, merged with an ensemble retriever).

This is the "framework" companion to a from-scratch RAG implementation — it shows how
LangChain's abstractions (`ChatGroq`, `ChatPromptTemplate`, `Chroma`,
`RecursiveCharacterTextSplitter`, `as_retriever`, `EnsembleRetriever`) replace the
manual plumbing (raw API calls, manual chunking, manual cosine similarity) with
composable building blocks.

## What's inside

| File | Description |
|---|---|
| `langchain_rag_hyde_hybrid.ipynb` | Full notebook, in four progressive parts |
| `requirements.txt` | Python dependencies |
| `NOTES.docx` (generated separately) | Line-by-line explanation of every cell |

## Structure of the notebook

### Part 1 — LangChain fundamentals
- Calling an LLM (`ChatGroq`) directly with `.invoke()`
- Prompt templates (`ChatPromptTemplate`) with placeholders
- **LCEL (LangChain Expression Language)** — chaining components with the `|`
  operator: `prompt | llm | StrOutputParser()`
- Why `StrOutputParser()` matters: it converts a structured `AIMessage` object into a
  plain string so it can be piped into further steps

### Part 2 — Standard RAG with LangChain
- PDF extraction with PyMuPDF (same as the from-scratch version)
- `RecursiveCharacterTextSplitter` — a smarter chunker than fixed-width slicing: it
  tries to split on paragraph breaks first, then lines, then spaces, so chunks respect
  natural text boundaries instead of cutting mid-sentence
- `Chroma.from_texts()` + `HuggingFaceEmbeddings` — LangChain's wrapper around
  ChromaDB and Sentence-Transformers
- `.as_retriever()` — replaces manually calling `collection.query()`; returns
  `Document` objects (`.page_content`) instead of raw strings
- Retriever search types compared: `similarity` (plain cosine similarity), `mmr`
  (Maximal Marginal Relevance — picks diverse results, avoids near-duplicate chunks),
  and `similarity_score_threshold` (only returns chunks above a confidence score)

### Part 3 — HyDE with LangChain
- A second prompt chain (`hyde_chain`) that asks the LLM to generate a detailed
  hypothetical document answering the question
- That hypothetical document — not the raw question — is embedded and used to query
  the vector store
- The final answer is still generated from the *real* retrieved chunks, not the
  hypothetical one — HyDE only changes what you search **with**, not what you answer
  **from**

### Part 4 — Hybrid Search (BM25 + Vector, via EnsembleRetriever)
- `BM25Retriever` — classic keyword-based retrieval (term frequency / inverse
  document frequency); good at exact keyword matches
- `Chroma.as_retriever()` — semantic/vector retrieval; good at meaning-based matches
- `EnsembleRetriever` — merges both result lists using **Reciprocal Rank Fusion
  (RRF)**, combining the strengths of keyword and semantic search
- Weighted 50/50 between BM25 and vector search in this notebook

## Interview-ready summary

> "I used LangChain to build three RAG variants on top of the same PDF pipeline:
> a standard vector-retrieval RAG using Chroma and HuggingFace embeddings, a HyDE
> variant where the LLM first generates a hypothetical answer that's embedded for
> retrieval instead of the raw query, and a hybrid retriever that fuses BM25
> keyword search with vector search using an EnsembleRetriever and Reciprocal
> Rank Fusion. I used LCEL (prompt | llm | parser) to compose each pipeline as a
> chain."

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You will also need:
- A PDF file to test on (set the path in `extract_text("your_file.pdf")`)
- A [Groq API key](https://console.groq.com/keys), set as an environment variable:
  ```bash
  export GROQ_API_KEY="your-key-here"
  ```
  and in the notebook use `ChatGroq(api_key=os.environ["GROQ_API_KEY"], model=...)`.

> **Security note:** the original notebook has `api_key=""` hardcoded in several
> cells. Never commit real API keys to GitHub — use environment variables or a
> `.env` file (add `.env` to `.gitignore`).

## Device note

Embeddings are loaded with `model_kwargs={"device": "mps"}`, which is Apple
Silicon's GPU backend. On Windows/Linux with an NVIDIA GPU, use `"cuda"`; on any
machine without a GPU, use `"cpu"`.

## Running

```bash
jupyter notebook langchain_rag_hyde_hybrid.ipynb
```

Run the notebook top to bottom. The `!pip install ...` cells at the top of the
notebook are Jupyter-only conveniences — if you're using `requirements.txt`, you can
skip those cells since the packages are already installed.

## Tech stack

- [LangChain](https://python.langchain.com/) — `langchain-core`, `langchain-groq`,
  `langchain-community`, `langchain-chroma`, `langchain-huggingface`,
  `langchain-classic`, `langchain-text-splitters`
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) — PDF text extraction
- [ChromaDB](https://www.trychroma.com/) — vector database
- [Sentence-Transformers](https://www.sbert.net/) via `HuggingFaceEmbeddings`
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — keyword retrieval
- [Groq](https://groq.com/) — fast LLM inference (`openai/gpt-oss-120b`,
  `llama-3.3-70b-versatile`)
