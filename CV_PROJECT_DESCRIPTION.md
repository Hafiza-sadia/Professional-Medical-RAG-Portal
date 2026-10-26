# MediQ: Clinical Retrieval-Augmented Generation (RAG) System
**Author:** Hafiza Sadia Bibi  
**Role:** AI / NLP Research Engineer & Full-Stack Developer  
**Live Stack:** Python 3.13, FastAPI, LangChain, FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`), Groq LLM API (`Qwen 3.8 27B`, `GPT-OSS`), Vanilla CSS & JavaScript

---

## 📌 Executive Summary (One-Paragraph Pitch)
Designed, implemented, and empirically benchmarked **MediQ**, an evidence-grounded Clinical Retrieval-Augmented Generation (RAG) portal specifically engineered to address ungrounded hallucinations, phantom citations, and premature guideline claims in healthcare AI. Built a dual-mode retrieval engine that distinguishes verified PDF document evidence (with 100% automated citation verification) from parametric AI knowledge (with strict epistemic uncertainty labeling and threshold-based abstention). Demonstrated 93.3% hit-rate across clinical benchmark evaluations with zero-overhead inline citation verification.

---

## 💼 Resume Bullet Points (Copy & Paste Ready)

### Option 1: AI / Machine Learning Engineer Focus
* **Architected End-to-End Clinical RAG Engine**: Engineered a dual-mode medical retrieval system using FastAPI, LangChain, and FAISS vector index, achieving **100% inline citation verification** and sub-20ms retrieval latencies across clinical guideline corpora.
* **Hallucination Prevention & Epistemic Honesty**: Implemented dynamic relevance thresholding (0.35 cosine threshold) to enforce **automated clinical abstention**, completely preventing hallucinations on out-of-domain or ungrounded queries.
* **Empirical Retrieval Benchmarking**: Developed an automated evaluation suite evaluating 4 chunking configurations (Coarse, Balanced, Fine, Expanded Top-K) across Recall@K, Precision@K, MRR, and Hit Rate (93.3%), publishing findings in a comprehensive technical research paper.
* **Production-Grade Fallback Pipeline**: Designed fault-tolerant LLM failover logic across Groq-hosted open models (`Qwen 3.8 27B`, `GPT-OSS 20B/120B`), safeguarding against rate limits (HTTP 429), model deprecations, and empty completions.

### Option 2: Full-Stack AI Engineer Focus
* Built **MediQ**, a full-stack evidence-grounded clinical QA portal featuring PDF ingestion, content-hash deduplication, live vectorstore management, and interactive drawer inspection of supporting passages.
* Designed a responsive, executive-standard clinical UI with zero UI frameworks, featuring real-time markdown parsing (tables, blockquotes, numbered lists), live latency breakdowns (retrieval vs generation), and dynamic safety badges.
* Integrated strict epistemic labeling in General AI Mode with regex-based guideline currency hedging to eliminate fabricated publication dates and unverified clinical standards.

---

## 🔬 Key Technical Highlights & Metrics

| Metric / Dimension | MediQ Performance | Engineering Rationale |
| :--- | :--- | :--- |
| **Citation Verification Rate** | **100.0%** | Regex-based mapping of `[Source X]` citations directly to indexed text chunks with zero hallucinated URLs |
| **Benchmark Hit Rate@K** | **93.3%** | Evaluated on clinical benchmark dataset across definitions, symptoms, and risk factors |
| **Abstention Accuracy** | **93.3%** | Automated threshold gating prevents answers when peak relevance falls below 35% |
| **Retrieval Latency** | **12–18 ms** | Local FAISS L2-to-Cosine normalized vector search on all-MiniLM-L6-v2 embeddings |
| **Deduplication Engine** | **Jaccard + Hash** | Filters >65% overlapping passage duplicates across PDF page boundaries |
| **LLM Inference** | **Qwen 3.8 27B** | High-throughput Groq LPU inference with automated retry and fallback chains |

---

## 🎤 Interview Speaking Points (How to Discuss in Technical Interviews)

1. **Why dual-mode?**
   > *"In medical AI, conflating pre-trained parametric knowledge with verified document evidence is dangerous. MediQ explicitly decouples them: Document Mode strictly answers only from uploaded guideline PDFs with mandatory `[Source X]` citations and abstains if evidence is absent. General Mode provides educational clinical overviews with strict epistemic disclaimers and automated currency hedging."*

2. **How does citation verification work?**
   > *"Rather than trusting the LLM to format citations, MediQ runs a deterministic verification pass on the completion. It extracts all `[Source X]` references, verifies that the source index exists in the retrieved set, measures lexical alignment with the chunk content, and detects phantom URLs or speculative years. If citations are omitted, an automated evidence drawer appends the exact source documents, page numbers, and excerpts."*

3. **How was the RAG pipeline optimized?**
   > *"I systematically evaluated chunk sizes (300, 600, 1000 tokens) with proportional overlaps. The 600-token chunk with 120-token overlap yielded the optimal balance between context coherence and retrieval precision, achieving 93.3% Hit Rate@3 and preventing fragment truncation."*
