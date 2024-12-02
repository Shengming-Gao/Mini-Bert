#!/usr/bin/env python3

"""
Multitask BERT Pretraining and Fine-Tuning Script.

This script performs the following:
1. Pretrains a BERT model using Masked Language Modeling (MLM) on the combined SST, Quora, and SemEval datasets.
2. Fine-tunes the pretrained model on three specific tasks:
   - Sentiment Classification
   - Paraphrase Detection
   - Semantic Textual Similarity
3. Evaluates the model using provided evaluation utility functions.
4. Generates predictions for test datasets.

**Note:** This script uses custom BERT implementations and does not rely on the `transformers` library.
Ensure that your custom modules (`bert`, `tokenizer`, `datasets`, `evaluation`, etc.) are correctly implemented.
"""

import time
import random
import numpy as np
import argparse
import sys
import re
import os
from types import SimpleNamespace
import csv

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import your custom BertModel and BertTokenizer
# Ensure that these are correctly implemented in your 'bert' and 'tokenizer' modules
from bert import BertModel
from tokenizer import BertTokenizer

# If you have a custom optimizer, ensure it's correctly implemented
# Otherwise, use PyTorch's built-in AdamW
try:
    from optimizer import AdamW
except ImportError:
    from torch.optim import AdamW

from tqdm import tqdm

# Import your dataset and evaluation utility functions
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    load_multitask_test_data
)

from evaluation import (
    model_eval_sst,
    model_eval_multitask,
    test_model_multitask
)

# Disable tqdm progress bars if needed
TQDM_DISABLE = False

# Fix the random seed for reproducibility
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Constants
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5  # Adjust based on your dataset

# MultitaskBERT Model with MLM Head
class MultitaskBERT(nn.Module):
    '''
    This module uses BERT for 4 tasks:

    - Masked Language Modeling (MLM)
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config, tokenizer):
        super(MultitaskBERT, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # Adjust as per your implementation

        # Freeze or unfreeze BERT parameters based on training phase
        if config.phase in ['pretrain', 'finetune', 'both']:
            for param in self.bert.parameters():
                param.requires_grad = True  # Trainable during pretraining and fine-tuning
        else:
            for param in self.bert.parameters():
                param.requires_grad = False  # Freeze all during evaluation or other phases

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Sentiment classification layer
        self.sentiment_classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Paraphrase detection layer
        self.paraphrase_classifier = nn.Linear(config.hidden_size * 2, 1)

        # Semantic similarity layer
        self.similarity_regressor = nn.Linear(config.hidden_size * 2, 1)

        # MLM head
        self.mlm_head = nn.Linear(config.hidden_size, tokenizer.vocab_size)
        self.tokenizer = tokenizer  # Store tokenizer for MLM

        # Tie weights between MLM head and BERT word embeddings
        self.mlm_head.weight = self.bert.word_embedding.weight  # Ensure this matches your BertModel implementation

        # Initialize weights for classification heads
        nn.init.xavier_uniform_(self.sentiment_classifier.weight)
        nn.init.zeros_(self.sentiment_classifier.bias)
        nn.init.xavier_uniform_(self.paraphrase_classifier.weight)
        nn.init.zeros_(self.paraphrase_classifier.bias)
        nn.init.xavier_uniform_(self.similarity_regressor.weight)
        nn.init.zeros_(self.similarity_regressor.bias)

    def forward(self, input_ids, attention_mask, task=None, labels=None, input_ids2=None, attention_mask2=None):
        '''
        General forward method that can handle different tasks.
        '''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']  # (batch_size, hidden_size)
        sequence_output = outputs['last_hidden_state']  # (batch_size, seq_len, hidden_size)
        pooled_output = self.dropout(pooled_output)

        if task == 'sentiment':
            logits = self.sentiment_classifier(pooled_output)
            loss = nn.CrossEntropyLoss()(logits, labels.view(-1))
            return loss, logits

        elif task == 'paraphrase':
            if input_ids2 is None or attention_mask2 is None:
                raise ValueError("For paraphrase task, input_ids2 and attention_mask2 must be provided.")
            outputs2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)
            pooled_output2 = outputs2['pooler_output']
            combined_pooled = torch.cat((pooled_output, pooled_output2), dim=1)  # (batch_size, hidden_size * 2)
            logits = self.paraphrase_classifier(combined_pooled)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.view(-1), labels.view(-1))
            return loss, logits

        elif task == 'similarity':
            if input_ids2 is None or attention_mask2 is None:
                raise ValueError("For similarity task, input_ids2 and attention_mask2 must be provided.")
            outputs2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)
            pooled_output2 = outputs2['pooler_output']
            combined_pooled = torch.cat((pooled_output, pooled_output2), dim=1)  # (batch_size, hidden_size * 2)
            logits = self.similarity_regressor(combined_pooled)
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.view(-1), labels.view(-1))
            return loss, logits

        elif task == 'mlm':
            prediction_scores = self.mlm_head(sequence_output)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape to (batch_size * seq_len, vocab_size)
            prediction_scores = prediction_scores.view(-1, self.mlm_head.weight.size(1))
            # Reshape labels to (batch_size * seq_len)
            labels = labels.view(-1)
            loss = loss_fct(prediction_scores, labels)
            return loss, prediction_scores

        else:
            raise ValueError(f"Unknown task: {task}")

    def predict_sentiment(self, input_ids, attention_mask):
        '''Sentiment prediction without loss computation'''
        with torch.no_grad():
            loss, logits = self.forward(input_ids, attention_mask, task='sentiment', labels=None)
        return logits

    def predict_paraphrase(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        '''Paraphrase prediction without loss computation'''
        with torch.no_grad():
            loss, logits = self.forward(input_ids=input_ids1, attention_mask=attention_mask1, task='paraphrase',
                                        labels=None, input_ids2=input_ids2, attention_mask2=attention_mask2)
        return torch.sigmoid(logits)

    def predict_similarity(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        '''Similarity prediction without loss computation'''
        with torch.no_grad():
            loss, logits = self.forward(input_ids=input_ids1, attention_mask=attention_mask1, task='similarity',
                                        labels=None, input_ids2=input_ids2, attention_mask2=attention_mask2)
        return logits.squeeze(-1)

# MLMDataset Class
class MLMDataset(Dataset):
    def __init__(self, input_ids, attention_mask, tokenizer, mlm_probability=0.15):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.labels = self.mask_tokens(input_ids)

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        # Create mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% MASK, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        labels[indices_replaced] = mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        labels[indices_random] = random_words[indices_random]

        # The rest 10% are left unchanged

        return labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }

# Save Model Function
def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': vars(args),
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saved the model to {filepath}")

# Data Extraction Functions
def extract_sentences_sst(file_path):
    import pandas as pd
    try:
        # Specify only the 'sentence' column to prevent parsing errors
        df = pd.read_csv(file_path, sep='\t', usecols=['sentence'])
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Ensure that the SST dataset at {file_path} has a 'sentence' column.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ParserError while reading {file_path}: {e}")
        sys.exit(1)

    return df['sentence'].dropna().tolist()

def extract_sentences_quora(file_path):
    import pandas as pd
    try:
        # Specify only the 'sentence1' and 'sentence2' columns
        df = pd.read_csv(file_path, sep='\t', usecols=['sentence1', 'sentence2'])
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Ensure that the Quora dataset at {file_path} has 'sentence1' and 'sentence2' columns.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ParserError while reading {file_path}: {e}")
        sys.exit(1)

    # Combine questions into a single list
    sentences = df['sentence1'].dropna().tolist() + df['sentence2'].dropna().tolist()
    return sentences

def extract_sentences_semeval(file_path):
    import pandas as pd
    try:
        # Specify only the 'sentence1' and 'sentence2' columns
        df = pd.read_csv(file_path, sep='\t', usecols=['sentence1', 'sentence2'])
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Ensure that the SemEval dataset at {file_path} has 'sentence1' and 'sentence2' columns.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ParserError while reading {file_path}: {e}")
        sys.exit(1)

    # Combine sentences into a single list
    sentences = df['sentence1'].dropna().tolist() + df['sentence2'].dropna().tolist()
    return sentences

# Prepare MLM Dataset
def prepare_mlm_data(args, tokenizer):
    '''
    Prepare MLM data by extracting sentences from SST, Quora, and SemEval datasets.
    '''
    # Extract sentences
    sst_train_sentences = extract_sentences_sst(args.sst_train)
    sst_dev_sentences = extract_sentences_sst(args.sst_dev)

    quora_train_sentences = extract_sentences_quora(args.para_train)
    quora_dev_sentences = extract_sentences_quora(args.para_dev)

    semeval_train_sentences = extract_sentences_semeval(args.sts_train)
    semeval_dev_sentences = extract_sentences_semeval(args.sts_dev)

    # Combine all sentences
    all_sentences = sst_train_sentences + sst_dev_sentences + \
                    quora_train_sentences + quora_dev_sentences + \
                    semeval_train_sentences + semeval_dev_sentences

    # Remove duplicates
    all_sentences = list(set(all_sentences))

    print(f"Total unique sentences for MLM pretraining: {len(all_sentences)}")

    # Tokenize
    inputs = tokenizer(all_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Create MLM dataset
    mlm_dataset = MLMDataset(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability
    )

    return mlm_dataset

# Pretraining Function
def pretrain_mlm(args, model, tokenizer, device, config):
    '''
    Pretrain the model using Masked Language Modeling (MLM) on the combined datasets.
    '''
    from torch.utils.data import DataLoader

    # Prepare MLM data
    mlm_dataset = prepare_mlm_data(args, tokenizer)
    mlm_dataloader = DataLoader(mlm_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=MLMDataset.collate_fn)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_mlm_loss = float('inf')

    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(mlm_dataloader, desc=f"Pretraining Epoch {epoch + 1}", disable=TQDM_DISABLE):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, task='mlm', labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Pretraining Epoch {epoch + 1}: Average MLM Loss = {avg_loss:.4f}")

        # Save the best model
        if avg_loss < best_mlm_loss:
            best_mlm_loss = avg_loss
            save_model(model, optimizer, args, config, args.mlm_save_path)
            print(f"Best MLM model saved with loss {best_mlm_loss:.4f}")

# Fine-Tuning Function
def train_multitask(args, model, tokenizer, device, config):
    '''
    Fine-tune the pretrained model on sentiment, paraphrase, and similarity tasks.
    '''
    # Load data for all three tasks
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    # Create datasets and dataloaders for Sentiment Analysis
    sst_train_dataset = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_dataset = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_dataset, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_dataset.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_dataset.collate_fn)

    # Create datasets and dataloaders for Paraphrase Detection
    para_train_dataset = SentencePairDataset(para_train_data, args)
    para_dev_dataset = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_dataset, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_dataset.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_dataset, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_dataset.collate_fn)

    # Create datasets and dataloaders for Semantic Textual Similarity
    sts_train_dataset = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_dataset, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_dataset.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_dataset.collate_fn)

    # Initialize the optimizer for fine-tuning
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_dev_metric = -float('inf')

    for epoch in range(args.finetune_epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        # Create iterators for each dataloader
        sst_iter = iter(sst_train_dataloader)
        para_iter = iter(para_train_dataloader)
        sts_iter = iter(sts_train_dataloader)

        # Determine the number of steps per epoch based on the largest dataloader
        num_steps = max(len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader))

        for step in tqdm(range(num_steps), desc=f"Fine-Tuning Epoch {epoch + 1}", disable=TQDM_DISABLE):
            # Randomly select a task to train on
            task = random.choice(['sentiment', 'paraphrase', 'similarity'])

            if task == 'sentiment':
                try:
                    batch = next(sst_iter)
                except StopIteration:
                    sst_iter = iter(sst_train_dataloader)
                    batch = next(sst_iter)
                b_ids, b_mask, b_labels = batch['token_ids'], batch['attention_mask'], batch['labels']
                b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

                optimizer.zero_grad()
                loss, logits = model(input_ids=b_ids, attention_mask=b_mask, task='sentiment', labels=b_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            elif task == 'paraphrase':
                try:
                    batch = next(para_iter)
                except StopIteration:
                    para_iter = iter(para_train_dataloader)
                    batch = next(para_iter)
                b_ids1, b_mask1 = batch['token_ids_1'], batch['attention_mask_1']
                b_ids2, b_mask2 = batch['token_ids_2'], batch['attention_mask_2']
                b_labels = batch['labels']
                b_ids1, b_mask1 = b_ids1.to(device), b_mask1.to(device)
                b_ids2, b_mask2 = b_ids2.to(device), b_mask2.to(device)
                b_labels = b_labels.to(device).float()

                optimizer.zero_grad()
                loss, logits = model(input_ids=b_ids1, attention_mask=b_mask1, task='paraphrase',
                                     labels=b_labels, input_ids2=b_ids2, attention_mask2=b_mask2)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            elif task == 'similarity':
                try:
                    batch = next(sts_iter)
                except StopIteration:
                    sts_iter = iter(sts_train_dataloader)
                    batch = next(sts_iter)
                b_ids1, b_mask1 = batch['token_ids_1'], batch['attention_mask_1']
                b_ids2, b_mask2 = batch['token_ids_2'], batch['attention_mask_2']
                b_labels = batch['labels']
                b_ids1, b_mask1 = b_ids1.to(device), b_mask1.to(device)
                b_ids2, b_mask2 = b_ids2.to(device), b_mask2.to(device)
                b_labels = b_labels.to(device).float()

                optimizer.zero_grad()
                loss, logits = model(input_ids=b_ids1, attention_mask=b_mask1, task='similarity',
                                     labels=b_labels, input_ids2=b_ids2, attention_mask2=b_mask2)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Evaluate the model on development data
        dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(
            sst_dev_dataloader,
            para_dev_dataloader,
            sts_dev_dataloader,
            model,
            device
        )

        # Aggregate the metrics; here we take the average
        dev_metric = (dev_sentiment_accuracy + dev_paraphrase_accuracy + dev_sts_corr) / 3

        if dev_metric > best_dev_metric:
            best_dev_metric = dev_metric
            save_model(model, optimizer, args, config, args.finetune_save_path)
            print(f"Best fine-tuned model saved with dev metric {best_dev_metric:.4f}")

        print(f"Epoch {epoch + 1}: Average Train Loss = {avg_train_loss:.3f}, "
              f"Sentiment Acc = {dev_sentiment_accuracy:.3f}, "
              f"Paraphrase Acc = {dev_paraphrase_accuracy:.3f}, "
              f"STS Corr = {dev_sts_corr:.3f}")

# Test Model Function
def test_model(args):
    '''
    Load the fine-tuned model and generate predictions for the test datasets.
    '''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')

        # Load the fine-tuned model
        if not os.path.exists(args.finetune_save_path):
            raise FileNotFoundError(f"Fine-tuned model not found at {args.finetune_save_path}. Please run fine-tuning first.")

        saved = torch.load(args.finetune_save_path, map_location=device)
        config = saved['model_config']

        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust as per your implementation

        # Initialize the model
        model = MultitaskBERT(config, tokenizer)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        model.eval()
        print(f"Loaded fine-tuned model from {args.finetune_save_path}")

        # Load test data
        sst_test_data, num_labels, para_test_data, sts_test_data = load_multitask_test_data(
            args.sst_test, args.para_test, args.sts_test)

        # Create datasets and dataloaders for testing
        sst_test_dataset = SentenceClassificationTestDataset(sst_test_data, args)
        sst_test_dataloader = DataLoader(sst_test_dataset, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=sst_test_dataset.collate_fn)

        para_test_dataset = SentencePairTestDataset(para_test_data, args)
        para_test_dataloader = DataLoader(para_test_dataset, shuffle=False, batch_size=args.batch_size,
                                          collate_fn=para_test_dataset.collate_fn)

        sts_test_dataset = SentencePairTestDataset(sts_test_data, args, isRegression=True)
        sts_test_dataloader = DataLoader(sts_test_dataset, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=sts_test_dataset.collate_fn)

        print("Starting model testing...")
        test_model_multitask(args, model, device,
                             sst_test_dataloader, para_test_dataloader, sts_test_dataloader)
        print("Model testing completed.")

# Argument Parsing Function
def get_args():
    parser = argparse.ArgumentParser(description="Multitask BERT Pretraining and Fine-Tuning")

    # Data paths
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv",
                        help="Path to the SST training data file.")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv",
                        help="Path to the SST development data file.")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv",
                        help="Path to the SST test data file.")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv",
                        help="Path to the Quora training data file.")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv",
                        help="Path to the Quora development data file.")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv",
                        help="Path to the Quora test data file.")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv",
                        help="Path to the SemEval training data file.")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv",
                        help="Path to the SemEval development data file.")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv",
                        help="Path to the SemEval test data file.")

    # MLM data paths
    parser.add_argument("--mlm_save_path", type=str, default="pretrained-mlm-model.pt",
                        help="Path to save the pretrained MLM model.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Probability of masking tokens for MLM pretraining.")

    # Fine-tuning save path
    parser.add_argument("--finetune_save_path", type=str, default="finetuned-model.pt",
                        help="Path to save the fine-tuned model.")

    # Training configurations
    parser.add_argument("--seed", type=int, default=11711,
                        help="Random seed for reproducibility.")
    parser.add_argument("--pretrain_epochs", type=int, default=5,
                        help="Number of epochs for MLM pretraining.")
    parser.add_argument("--finetune_epochs", type=int, default=10,
                        help="Number of epochs for fine-tuning.")

    # Flags for training phases
    parser.add_argument("--pretrain", action='store_true',
                        help="Flag to perform MLM pretraining before fine-tuning.")
    parser.add_argument("--finetune", action='store_true',
                        help="Flag to perform fine-tuning on specific tasks.")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3,
                        help="Dropout probability for hidden layers.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate.")

    # Device
    parser.add_argument("--use_gpu", action='store_true',
                        help="Use GPU for training.")

    # Output paths for predictions
    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv",
                        help="Output path for SST development predictions.")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv",
                        help="Output path for SST test predictions.")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv",
                        help="Output path for Paraphrase development predictions.")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv",
                        help="Output path for Paraphrase test predictions.")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv",
                        help="Output path for SemEval development predictions.")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv",
                        help="Output path for SemEval test predictions.")

    args = parser.parse_args()
    return args

# Main Execution Flow
if __name__ == "__main__":
    args = get_args()

    # Validate Training Phases
    if not args.pretrain and not args.finetune:
        print("Error: At least one of --pretrain or --finetune must be specified.")
        sys.exit(1)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust as per your implementation

    # Create model configuration
    if args.pretrain and args.finetune:
        phase = 'both'
    elif args.pretrain:
        phase = 'pretrain'
    elif args.finetune:
        phase = 'finetune'
    else:
        phase = 'none'  # Should not reach here due to earlier check

    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': N_SENTIMENT_CLASSES,  # Assuming consistent across tasks
        'hidden_size': BERT_HIDDEN_SIZE,
        'data_dir': '.',  # Adjust if necessary
        'phase': phase
    }
    config = SimpleNamespace(**config)

    # Initialize the model
    model = MultitaskBERT(config, tokenizer)

    # Set device
    device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # Fix the random seed for reproducibility
    seed_everything(args.seed)

    # Perform MLM Pretraining if flag is set
    if args.pretrain:
        print("Starting MLM Pretraining...")
        pretrain_mlm(args, model, tokenizer, device, config)
    else:
        print("Skipping MLM Pretraining.")

    # Perform Fine-Tuning if flag is set
    if args.finetune:
        # Load the pretrained MLM model if pretraining was performed in a previous run
        if not args.pretrain:
            if not os.path.exists(args.mlm_save_path):
                raise FileNotFoundError(f"Pretrained MLM model not found at {args.mlm_save_path}. Please run pretraining first.")
            print(f"Loading pretrained MLM model from {args.mlm_save_path}...")
            saved_mlm = torch.load(args.mlm_save_path, map_location=device)
            model.load_state_dict(saved_mlm['model'])
            print("Pretrained MLM model loaded successfully.")

        print("Starting Fine-Tuning...")
        train_multitask(args, model, tokenizer, device, config)
    else:
        print("Skipping Fine-Tuning.")

    # Testing Phase (only if fine-tuning is done)
    if args.finetune:
        print("Starting Testing...")
        test_model(args)
        print("Model testing completed.")
    else:
        print("Skipping Testing Phase.")
