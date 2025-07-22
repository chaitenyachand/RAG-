# RAG System with Gemini and LangChain 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![LangChain](https://img.shields.io/badge/LangChain-b033d6?logo=langchain)](https://www.langchain.com/)
[![Gemini](https://img.shields.io/badge/Gemini-8e75b7?logo=google&logoColor=white)](https://ai.google.dev/models/gemini)

This repository contains a complete Retrieval-Augmented Generation (RAG) system built using Google's Gemini Pro, LangChain, and FAISS for efficient vector storage and retrieval. The project is designed to answer questions based on the content of a provided PDF document.

---

## 📋 Table of Contents
* [Overview](#-overview)
* [Workflow](#-workflow)
* [Tech Stack](#-tech-stack)
* [Setup and Installation](#-setup-and-installation)
* [Usage](#-usage)
* [License](#-license)

---

## 📝 Overview

This project implements a RAG pipeline that leverages large language models (LLMs) to provide answers to user queries from a custom knowledge base. Instead of relying on the LLM's pre-trained knowledge, it first retrieves relevant information from a specific document (in this case, a PDF) and then uses that context to generate a precise and factual answer.

### Key Features
- **PDF Data Source**: Ingests and processes text from PDF files.
- **Text Chunking**: Splits the document into smaller, manageable chunks for effective embedding.
- **Vector Embeddings**: Uses Google's embedding models to convert text chunks into vector representations.
- **FAISS Vector Store**: Stores and indexes vectors for fast and efficient similarity searches.
- **Gemini Pro Integration**: Utilizes Google's Gemini Pro model for generating answers based on the retrieved context.

---

## ⚙️ Workflow

The system follows a standard RAG architecture:

```mermaid
graph TD
    A[1. Load PDF] --> B[2. Split into Text Chunks];
    B --> C[3. Create Vector Embeddings];
    C --> D[4. Store in FAISS Vector DB];
    E[User Query] --> F[5. Embed Query];
    F --> G{6. Similarity Search in FAISS};
    G -- Retrieved Chunks --> H;
    D -- Knowledge Base --> G;
    E --> H{7. Augment Prompt};
    H --> I[8. Get Answer from Gemini Pro];
    I --> J[Final Answer];
