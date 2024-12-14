MiniBERT for Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity
This repository contains the implementation and experimentation of a simplified BERT model (MiniBERT) to solve three downstream NLP tasks: Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity (STS). The project follows the guidelines from the CS224N Final Project, with extensions and experiments designed to improve multi-task learning and fine-tuning.

GitHub Repository: Shengming-Gao/Mini-BERT

Table of Contents
Overview
Setup
Training and Evaluation
Pretraining
Finetuning
Evaluation
Implemented Features
Extensions
Results
Future Work
References
Overview
MiniBERT is a simplified implementation of the original BERT model focused on achieving robust contextual embeddings for:

Sentiment Analysis: Predicting sentiment labels for input sentences (SST Dataset).
Paraphrase Detection: Determining whether two sentences convey the same meaning (Quora Dataset).
Semantic Textual Similarity (STS): Measuring the similarity between two sentences on a scale of 0 to 5 (SemEval Dataset).
Key methodologies explored in this project:

Multi-task fine-tuning and pretraining.
Gradient surgery to reduce task conflicts.
Cosine similarity loss for STS fine-tuning.
Setup
Prerequisites
Python >= 3.8
PyTorch >= 1.11
Anaconda/Miniconda environment
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/Shengming-Gao/Mini-Bert.git
cd Mini-Bert
Setup the environment:
bash
Copy code
source setup.sh
conda activate cs224n_dfp
Install additional packages if needed:
bash
Copy code
pip install zipp==3.11.0 idna==3.4 chardet==4.0.0
Training and Evaluation
Pretraining
To pretrain the MiniBERT model on the classification heads for each task:

bash
Copy code
python3 multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu
Finetuning
To fine-tune the pre-trained model on multi-task objectives:

bash
Copy code
python3 multitask_classifier.py --option finetune --lr 1e-5 --use_gpu --pretrain_filepath "path_to_pretrained_model.pt"
Evaluation
To evaluate a pre-trained or fine-tuned model:

bash
Copy code
python3 evaluate_pretrained_model.py --use_gpu --filepath "path_to_model.pt"
Custom Hyperparameters
Modify additional hyperparameters like batch size, number of epochs, and dropout probability directly in the config.py file or by appending arguments to the commands:

bash
Copy code
--batch_size 16 --num_epochs 10 --hidden_dropout_prob 0.1
Implemented Features
BERT Transformer Implementation

Multi-head self-attention layer
Transformer encoder layers
Token and position embeddings
Downstream Task Classifiers

Sentiment Analysis (SST)
Paraphrase Detection (Quora)
Semantic Textual Similarity (SemEval)
Custom Optimizer

Adam optimizer with decoupled weight decay.
Multi-task Learning

Pretraining and fine-tuning strategies:
Sequential pretraining and interleaving fine-tuning.
Gradient surgery to resolve gradient conflicts between tasks.
Extended Loss Functions

Cosine similarity loss for STS fine-tuning.
Extensions
Gradient Surgery: Reduced gradient conflicts during multi-task training by projecting task-specific gradients onto the orthogonal space of conflicting gradients.
Cosine Similarity Fine-Tuning: Improved STS task performance by leveraging the cosine similarity between sentence embeddings during fine-tuning.
Regularization Techniques: Experimented with dropout and smoothness-inducing regularization to mitigate overfitting.
Results
The project achieved the best results using a combination of sequential pretraining and interleaving fine-tuning with integrated cosine similarity loss. Key metrics:

Sentiment Accuracy: 0.505
Paraphrase Accuracy: 0.788
STS Correlation: 0.772
Refer to the results table for detailed comparisons across iterations.

Future Work
Potential improvements and extensions:

Regularized Optimization: Implement smoothness-inducing adversarial regularization to prevent overfitting during fine-tuning.
Contrastive Learning: Explore contrastive loss frameworks like SimCSE for improving sentence embeddings.
Dynamic Sampling: Balance tasks dynamically during interleaving fine-tuning to address dataset imbalance.
References
Key papers and resources used in this project:

Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
Reimers & Gurevych, Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (2019)
Yu et al., Gradient Surgery for Multi-task Learning (2020)
