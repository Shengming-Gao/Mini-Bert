# MiniBERT for Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity

This repository contains the implementation and experimentation of a simplified BERT model (MiniBERT) to solve three downstream NLP tasks: **Sentiment Analysis**, **Paraphrase Detection**, and **Semantic Textual Similarity (STS)**. The project follows the guidelines from the CS224N Final Project, with extensions and experiments designed to improve multi-task learning and fine-tuning.

GitHub Repository: [Shengming-Gao/Mini-Bert](https://github.com/Shengming-Gao/Mini-Bert)

---

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Training and Evaluation](#training-and-evaluation)
  - [Pretraining](#pretraining)
  - [Finetuning](#finetuning)
  - [Evaluation](#evaluation)
- [Implemented Features](#implemented-features)
- [Extensions](#extensions)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

MiniBERT is a simplified implementation of the original BERT model focused on achieving robust contextual embeddings for:
- **Sentiment Analysis**: Predicting sentiment labels for input sentences (SST Dataset).
- **Paraphrase Detection**: Determining whether two sentences convey the same meaning (Quora Dataset).
- **Semantic Textual Similarity (STS)**: Measuring the similarity between two sentences on a scale of 0 to 5 (SemEval Dataset).

Key methodologies explored in this project:
- Multi-task fine-tuning and pretraining.
- Gradient surgery to reduce task conflicts.
- Cosine similarity loss for STS fine-tuning.

---

## Setup

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.11
- Anaconda/Miniconda environment

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Shengming-Gao/Mini-Bert.git
   cd Mini-Bert



## Training and Evaluation

### Pretraining
To pretrain the MiniBERT model on the classification heads for each task:

python3 multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu



### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Shengming-Gao/Mini-Bert.git
   cd Mini-Bert
