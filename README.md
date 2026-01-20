RAG-Based Financial & Legal Question Answering System
Overview

This project implements a Retrieval-Augmented Generation (RAG) system to answer complex financial and legal questions using Apple’s 2024 Form 10-K and Tesla’s 2023 Form 10-K filings.
The system uses open-source, locally hosted models only, ensuring transparency, reproducibility, and compliance with assignment constraints.

Key Features

PDF ingestion with page-level metadata

Semantic chunking and vector indexing

FAISS-based similarity search

Optional cross-encoder re-ranking

Local LLM answer generation with strict grounding

Source-cited, auditable responses


System Architecture

Document Ingestion

Parse PDFs into text

Preserve metadata (document name, section, page)

Chunking & Embedding

Split text into manageable chunks

Generate embeddings using an open-source model

Vector Indexing

Store embeddings in FAISS for fast similarity search

Retrieval (+ Re-ranking)

Retrieve top-k relevant chunks

Optionally re-rank using a cross-encoder for precision

LLM Answer Generation

Use a local instruction-tuned LLM

Enforce strict context usage and citation rules

Prompting Rules (Strictly Enforced)

Answers must use only retrieved context

Every factual claim must include a citation:

["Apple 10-K", "Item 8", "p.28"]


If information is missing:

Not specified in the document.


If the question is out of scope:

This question cannot be answered based on the provided documents.

How to Run
pip install -r requirements.txt
python main.py


Ensure all PDF files are placed inside the data/ directory.

Design Rationale

FAISS: Fast, local, production-proven vector search

Chunking: Prevents token overflow and improves retrieval quality

Re-ranking: Improves answer precision for financial/legal queries

Local LLM: Avoids reliance on closed or paid APIs

Limitations & Learnings

Token limits require careful context trimming

Metadata consistency is critical for citation accuracy

Local LLM inference is slower than cloud-based alternatives

Conclusion

This project demonstrates a grounded, auditable RAG pipeline suitable for enterprise financial and legal question answering, built entirely using open-source tools and best practices.
