
# Retrieval-Augmented Generation (RAG)

This repository provides an implementation of **Retrieval-Augmented Generation (RAG)** — a hybrid system that combines **information retrieval** with **large language models (LLMs)** to generate more accurate and grounded answers.

---

## How It Works

```mermaid
flowchart TD
    A["User Query"] --> B["Retriever"]
    B -->|BM25 / Dense / FAISS| C["Relevant Documents"]
    C --> D["Generator (LLM)"]
    D --> E["Final Answer"]
````

* **Retriever** → Finds relevant documents using BM25, Dense embeddings, or FAISS
* **Generator (LLM)** → Uses the retrieved documents to generate an informed response
* **Final Answer** → Output is factual, grounded, and context-aware

---

## Repository Structure

```mermaid
mindmap
  root((RAG Repo))
    Controller
      Features
      Train Controller
      Switcher
    Retrieval
      BM25
      Dense Retrieval
      FAISS
    Generation
      Generator
    Data
      HotpotQA
      Natural Questions
      SciQ
    Evaluation
      Evaluate
    Main Script
      main.py
```

---

## Usage

Run the main pipeline:

```bash
python main.py
```

---

## Installation

```bash
git clone <repo-url>
cd Retrieval-Augmented-Generation-main
pip install -r requirements.txt
```

---

## Module Guide

## Data

Prepare datasets (HotpotQA, NQ, SciQ):

```bash
python data/download_hotpotqa.py
python data/download_nq.py
```

Load custom subsets:

```bash
python data/hotpotqa_subset.py
```

---

### Retrieval

Test retrieval backends:

```bash
python retrieval/bm25.py
python retrieval/dense_retrieval.py
python retrieval/faiss_dense.py
```

---

### Generation

Generate answers from retrieved context:

```bash
python generation/generator.py
```

---

### Evaluation

Evaluate performance:

```bash
python evaluation/evaluate.py
```

---

## Requirements

* Python 3.8+
* Dependencies listed in `requirements.txt`

---

## License

This project is licensed under the MIT License.

