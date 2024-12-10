import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, evaluate_paraphrase, evaluate_sts


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class MultitaskBERT(nn.Module):
    '''
    This module uses BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze or unfreeze BERT parameters based on training option
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Sentiment classification layer
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)

        # Paraphrase detection layer
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

        # Semantic similarity layer
        self.similarity_regressor = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

        # Initialize weights
        nn.init.xavier_uniform_(self.sentiment_classifier.weight)
        nn.init.zeros_(self.sentiment_classifier.bias)
        nn.init.xavier_uniform_(self.paraphrase_classifier.weight)
        nn.init.zeros_(self.paraphrase_classifier.bias)
        nn.init.xavier_uniform_(self.similarity_regressor.weight)
        nn.init.zeros_(self.similarity_regressor.bias)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # Get the pooled output from BERT (hidden state of [CLS] token)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output'] # (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1 - somewhat negative, 2 - neutral, 3 - somewhat positive, 4 - positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        # Get sentence embeddings
        pooled_output = self.forward(input_ids, attention_mask)
        # Classify sentiment
        logits = self.sentiment_classifier(pooled_output)  # (batch_size, N_SENTIMENT_CLASSES)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # Get embeddings for both sentences
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        # Concatenate embeddings
        combined_output = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        combined_output = self.dropout(combined_output)
        # Predict paraphrase logit
        logits = self.paraphrase_classifier(combined_output).squeeze(-1)  # (batch_size)
        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        # Get embeddings for both sentences
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        # Concatenate embeddings
        combined_output = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        combined_output = self.dropout(combined_output)
        # Predict similarity logit
        logits = self.similarity_regressor(combined_output).squeeze(-1)  # (batch_size)
        return logits


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load train and dev data for all tasks
    sst_train_raw, num_labels, para_train_raw, sts_train_raw = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_raw, _, para_dev_raw, sts_dev_raw = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    # Create datasets and dataloaders for SST (classification)
    sst_train_data = SentenceClassificationDataset(sst_train_raw, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_raw, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Create datasets and dataloaders for Paraphrase (binary classification)
    # Note: SentencePairDataset expects tuples: (sent1, sent2, label, id)
    # para_train_raw and para_dev_raw are already loaded by load_multitask_data
    para_train_data = SentencePairDataset(para_train_raw, args, isRegression=False)
    para_dev_data = SentencePairDataset(para_dev_raw, args, isRegression=False)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    # Create datasets and dataloaders for STS (regression)
    sts_train_data = SentencePairDataset(sts_train_raw, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_raw, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'data_dir': '.',
        'option': args.option
    }

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training SST (Sentiment Classification)
    best_dev_acc = 0
    print("=== Training on SST (Sentiment) ===")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'SST-train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"[SST] Epoch {epoch}: train loss={train_loss:.3f}, train acc={train_acc:.3f}, dev acc={dev_acc:.3f}")

    # Load the best SST model before moving to next tasks
    saved = torch.load(args.filepath)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    # Training Paraphrase (Binary Classification)
    # Use BCEWithLogitsLoss for binary classification
    best_para_acc = 0
    print("=== Training on Quora Paraphrase (Binary Classification) ===")
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'Para-train-{epoch}', disable=TQDM_DISABLE):
            b1_ids = batch['token_ids_1'].to(device)
            b1_mask = batch['attention_mask_1'].to(device)
            b2_ids = batch['token_ids_2'].to(device)
            b2_mask = batch['attention_mask_2'].to(device)
            labels = batch['labels'].float().to(device)  # binary label 0 or 1

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b1_ids, b1_mask, b2_ids, b2_mask)
            loss = bce_loss_fn(logits, labels) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        # Evaluate on the paraphrase dev set
        # Todo: reference to evaluation.py and modify the following line.
        dev_acc, dev_f1 = evaluate_paraphrase(para_dev_dataloader, model, device)
        if dev_acc > best_para_acc:
            best_para_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"[Para] Epoch {epoch}: train loss={train_loss:.3f}, dev acc={dev_acc:.3f}")

    # Load the best Paraphrase model before moving on
    saved = torch.load(args.filepath)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    # Training STS (Regression)
    # Use MSELoss for regression
    best_sts_pearson = 0  # or any metric you want to track
    print("=== Training on STS (Regression) ===")
    mse_loss_fn = nn.MSELoss(reduction='sum')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sts_train_dataloader, desc=f'STS-train-{epoch}', disable=TQDM_DISABLE):
            b1_ids = batch['token_ids_1'].to(device)
            b1_mask = batch['attention_mask_1'].to(device)
            b2_ids = batch['token_ids_2'].to(device)
            b2_mask = batch['attention_mask_2'].to(device)
            labels = batch['labels'].float().to(device) # similarity scores

            optimizer.zero_grad()
            logits = model.predict_similarity(b1_ids, b1_mask, b2_ids, b2_mask)
            loss = mse_loss_fn(logits, labels) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        # Evaluate on the STS dev set
        # Todo: reference to evaluation.py and modify the following line.
        dev_pearson = evaluate_sts(sts_dev_dataloader, model, device)
        if dev_pearson > best_sts_pearson:
            best_sts_pearson = dev_pearson
            save_model(model, optimizer, args, config, args.filepath)

        print(f"[STS] Epoch {epoch}: train loss={train_loss:.3f}, dev pearson={dev_pearson:.3f}")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)

