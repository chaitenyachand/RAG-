# RAG++: A Hybrid Sparse-Dense Adaptive Retrieval-Augmented Generation System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

**RAG++** is a hybrid Retrieval-Augmented Generation (RAG) framework that adaptively selects between sparse (BM25) and dense (FAISS) retrieval methods using a trained controller. It improves performance for domain-specific and low-resource Question Answering (QA) tasks by dynamically choosing the optimal retriever for each query.

---

## Overview

Traditional Retrieval-Augmented Generation systems typically rely on a single, fixed retrieval strategy. RAG++ introduces an adaptive approach where a logistic regression-based controller learns to select the most suitable retriever—either sparse or dense—based on the linguistic and semantic features of the query. The documents retrieved by the selected method are then passed to a language model (such as T5-small or GPT-3.5) to generate a precise and contextually accurate answer.

---

## Core Features

- **Hybrid Retrieval**: Combines the strengths of sparse **BM25** (for keyword matching) and dense **FAISS** (for semantic similarity) retrievers.
- **Adaptive Controller**: Employs a machine learning model to dynamically select the best retriever on a per-query basis.
- **Modular Design**: Built for easy integration and extension, allowing components to be used in other QA pipelines.
- **Comprehensive Evaluation**: Assesses system performance using standard metrics including F1 Score, Exact Match (EM), and BLEU.
- **Dataset Compatibility**: Pre-configured for use with the **HotpotQA** and **SciQ** datasets.

---

## Installation Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/RAG-Plus-Plus.git](https://github.com/yourusername/RAG-Plus-Plus.git)
cd RAG-Plus-Plus

### 2. Set Up the Environment
It is recommended to use conda for dependency management to ensure a clean environment.

'''bash
conda create -n ragpp python=3.9
conda activate ragpp
pip install -r requirements.txt

## Usage
To run the complete training and evaluation pipeline, execute the main script from the root directory:

'''bash
python main.py
This command will initiate the following sequence:
