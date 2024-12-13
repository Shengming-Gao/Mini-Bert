import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
from datetime import datetime

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

TQDM_DISABLE = True

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def predict_sentiment(self, input_ids, attention_mask):
        pooled_output = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(pooled_output)
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        combined_output = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        combined_output = self.dropout(combined_output)
        logits = self.paraphrase_classifier(combined_output).squeeze(-1)
        return logits

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        combined_output = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        combined_output = self.dropout(combined_output)
        logits = self.similarity_regressor(combined_output).squeeze(-1)
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
    print(f"Model saved to {filepath}")

def pretrain_multitask_sequential(args, logger):
    """
    Pretrain stage (heads only, BERT frozen) done sequentially:
    1. Pretrain on SST
    2. Pretrain on Paraphrase
    3. Pretrain on STS
    """
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load train and dev data
    sst_train_raw, num_labels, para_train_raw, sts_train_raw = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_raw, _, para_dev_raw, sts_dev_raw = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    # Datasets and Dataloaders
    sst_train_data = SentenceClassificationDataset(sst_train_raw, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_raw, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_raw, args, isRegression=False)
    para_dev_data = SentencePairDataset(para_dev_raw, args, isRegression=False)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_raw, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_raw, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Model config
    pretrain_config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'data_dir': '.',
        'option': 'pretrain'  # ensure pretrain mode
    }
    pretrain_config = SimpleNamespace(**pretrain_config)
    model = MultitaskBERT(pretrain_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Pretrain on SST
    logger.write("=== Pretraining on SST (Sentiment) ===\n")
    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'SST-pretrain-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        train_acc, _, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, _, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, pretrain_config, args.pretrain_filepath)

        logger.write(f"[SST-pretrain] Epoch {epoch}: train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, dev_acc={dev_acc:.3f}\n")

    # Load the best SST model
    saved = torch.load(args.pretrain_filepath)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    # Pretrain on Paraphrase (heads only)
    logger.write("=== Pretraining on Quora Paraphrase ===\n")
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    best_para_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'Para-pretrain-{epoch}', disable=TQDM_DISABLE):
            b1_ids = batch['token_ids_1'].to(device)
            b1_mask = batch['attention_mask_1'].to(device)
            b2_ids = batch['token_ids_2'].to(device)
            b2_mask = batch['attention_mask_2'].to(device)
            labels = batch['labels'].float().to(device)
            optimizer.zero_grad()
            logits = model.predict_paraphrase(b1_ids, b1_mask, b2_ids, b2_mask)
            loss = bce_loss_fn(logits, labels) / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        dev_acc, _ = evaluate_paraphrase(para_dev_dataloader, model, device)
        if dev_acc > best_para_acc:
            best_para_acc = dev_acc
            save_model(model, optimizer, args, pretrain_config, args.pretrain_filepath)

        logger.write(f"[Para-pretrain] Epoch {epoch}: train_loss={train_loss:.3f}, dev_acc={dev_acc:.3f}\n")

    # Load the best model so far (with SST and Paraphrase heads)
    saved = torch.load(args.pretrain_filepath)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    # Pretrain on STS (heads only)
    logger.write("=== Pretraining on STS (Regression) ===\n")
    mse_loss_fn = nn.MSELoss(reduction='sum')
    best_sts_pearson = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sts_train_dataloader, desc=f'STS-pretrain-{epoch}', disable=TQDM_DISABLE):
            b1_ids = batch['token_ids_1'].to(device)
            b1_mask = batch['attention_mask_1'].to(device)
            b2_ids = batch['token_ids_2'].to(device)
            b2_mask = batch['attention_mask_2'].to(device)
            labels = batch['labels'].float().to(device)
            optimizer.zero_grad()
            logits = model.predict_similarity(b1_ids, b1_mask, b2_ids, b2_mask)
            loss = mse_loss_fn(logits, labels) / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        dev_pearson = evaluate_sts(sts_dev_dataloader, model, device)
        if dev_pearson > best_sts_pearson:
            best_sts_pearson = dev_pearson
            save_model(model, optimizer, args, pretrain_config, args.pretrain_filepath)

        logger.write(f"[STS-pretrain] Epoch {epoch}: train_loss={train_loss:.3f}, dev_pearson={dev_pearson:.3f}\n")

    logger.write("Pretraining finished.\n")


# def finetune_multitask_interleaving(args, logger):
#     """
#     Finetune stage (BERT and heads) done interleavingly:
#     """
#     device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
#
#     # Load data
#     sst_train_raw, num_labels, para_train_raw, sts_train_raw = load_multitask_data(
#         args.sst_train, args.para_train, args.sts_train, split='train')
#     sst_dev_raw, _, para_dev_raw, sts_dev_raw = load_multitask_data(
#         args.sst_dev, args.para_dev, args.sts_dev, split='train')
#
#     sst_train_data = SentenceClassificationDataset(sst_train_raw, args)
#     para_train_data = SentencePairDataset(para_train_raw, args, isRegression=False)
#     sts_train_data = SentencePairDataset(sts_train_raw, args, isRegression=True)
#
#     sst_dev_data = SentenceClassificationDataset(sst_dev_raw, args)
#     para_dev_data = SentencePairDataset(para_dev_raw, args, isRegression=False)
#     sts_dev_data = SentencePairDataset(sts_dev_raw, args, isRegression=True)
#
#     sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
#                                       collate_fn=sst_train_data.collate_fn)
#     para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
#                                        collate_fn=para_train_data.collate_fn)
#     sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
#                                       collate_fn=sts_train_data.collate_fn)
#
#     sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
#                                     collate_fn=sst_dev_data.collate_fn)
#     para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
#                                      collate_fn=para_dev_data.collate_fn)
#     sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
#                                     collate_fn=sts_dev_data.collate_fn)
#
#     finetune_config = {
#         'hidden_dropout_prob': args.hidden_dropout_prob,
#         'num_labels': num_labels,
#         'hidden_size': 768,
#         'data_dir': '.',
#         'option': 'finetune'
#     }
#     finetune_config = SimpleNamespace(**finetune_config)
#     model = MultitaskBERT(finetune_config).to(device)
#     optimizer = AdamW(model.parameters(), lr=args.lr)
#
#     # Load pretrained model weights
#     saved = torch.load(args.pretrain_filepath)
#     model.load_state_dict(saved['model'])
#     model = model.to(device)
#
#     bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
#     mse_loss_fn = nn.MSELoss(reduction='sum')
#
#     best_sst_acc = 0
#     best_para_acc = 0
#     best_sts_pearson = 0
#
#     sst_iter = iter(sst_train_dataloader)
#     para_iter = iter(para_train_dataloader)
#     sts_iter = iter(sts_train_dataloader)
#
#     num_steps = max(len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader))
#     tasks = ['sst', 'para', 'sts']  # round robin
#
#     logger.write("=== Finetuning Interleavingly ===\n")
#     for epoch in range(args.epochs):
#         model.train()
#         sst_iter = iter(sst_train_dataloader)
#         para_iter = iter(para_train_dataloader)
#         sts_iter = iter(sts_train_dataloader)
#
#         train_loss = 0.0
#         num_batches = 0
#         for step in range(num_steps * len(tasks)):
#             task = tasks[step % len(tasks)]
#             if task == 'sst':
#                 try:
#                     batch = next(sst_iter)
#                 except StopIteration:
#                     continue
#                 b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
#                 optimizer.zero_grad()
#                 logits = model.predict_sentiment(b_ids, b_mask)
#                 loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
#
#             elif task == 'para':
#                 try:
#                     batch = next(para_iter)
#                 except StopIteration:
#                     continue
#                 b1_ids = batch['token_ids_1'].to(device)
#                 b1_mask = batch['attention_mask_1'].to(device)
#                 b2_ids = batch['token_ids_2'].to(device)
#                 b2_mask = batch['attention_mask_2'].to(device)
#                 labels = batch['labels'].float().to(device)
#                 optimizer.zero_grad()
#                 logits = model.predict_paraphrase(b1_ids, b1_mask, b2_ids, b2_mask)
#                 loss = bce_loss_fn(logits, labels) / args.batch_size
#
#             else:  # sts
#                 try:
#                     batch = next(sts_iter)
#                 except StopIteration:
#                     continue
#                 b1_ids = batch['token_ids_1'].to(device)
#                 b1_mask = batch['attention_mask_1'].to(device)
#                 b2_ids = batch['token_ids_2'].to(device)
#                 b2_mask = batch['attention_mask_2'].to(device)
#                 labels = batch['labels'].float().to(device)
#                 optimizer.zero_grad()
#                 logits = model.predict_similarity(b1_ids, b1_mask, b2_ids, b2_mask)
#                 loss = mse_loss_fn(logits, labels) / args.batch_size
#
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             num_batches += 1
#
#         train_loss = train_loss / (num_batches if num_batches > 0 else 1)
#         sst_dev_acc, _, *_ = model_eval_sst(sst_dev_dataloader, model, device)
#         para_dev_acc, _ = evaluate_paraphrase(para_dev_dataloader, model, device)
#         sts_dev_pearson = evaluate_sts(sts_dev_dataloader, model, device)
#
#         improved = False
#         if sst_dev_acc > best_sst_acc:
#             best_sst_acc = sst_dev_acc
#             improved = True
#         if para_dev_acc > best_para_acc:
#             best_para_acc = para_dev_acc
#             improved = True
#         if sts_dev_pearson > best_sts_pearson:
#             best_sts_pearson = sts_dev_pearson
#             improved = True
#
#         if improved:
#             save_model(model, optimizer, args, finetune_config, args.finetune_filepath)
#
#         logger.write(f"[Finetune] Epoch {epoch}: train_loss={train_loss:.3f}, SST_dev_acc={sst_dev_acc:.3f}, Para_dev_acc={para_dev_acc:.3f}, STS_dev_pearson={sts_dev_pearson:.3f}\n")
#
#     logger.write("Finetuning finished.\n")

def finetune_multitask_interleaving(args, logger):
    """
    Finetune stage (BERT and heads) done interleavingly with Gradient Surgery:
    We will:
    - Sample batches from each task (SST, Para, STS)
    - Compute losses and gradients separately
    - Apply Gradient Surgery to handle conflicting gradients
    - Then perform an optimizer step
    """
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load data
    sst_train_raw, num_labels, para_train_raw, sts_train_raw = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_raw, _, para_dev_raw, sts_dev_raw = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_raw, args)
    para_train_data = SentencePairDataset(para_train_raw, args, isRegression=False)
    sts_train_data = SentencePairDataset(sts_train_raw, args, isRegression=True)

    sst_dev_data = SentenceClassificationDataset(sst_dev_raw, args)
    para_dev_data = SentencePairDataset(para_dev_raw, args, isRegression=False)
    sts_dev_data = SentencePairDataset(sts_dev_raw, args, isRegression=True)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)

    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    finetune_config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'data_dir': '.',
        'option': 'finetune'
    }
    finetune_config = SimpleNamespace(**finetune_config)
    model = MultitaskBERT(finetune_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Load pretrained model weights
    saved = torch.load(args.pretrain_filepath, map_location=device)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    mse_loss_fn = nn.MSELoss(reduction='sum')

    best_sst_acc = 0
    best_para_acc = 0
    best_sts_pearson = 0

    sst_iter = iter(sst_train_dataloader)
    para_iter = iter(para_train_dataloader)
    sts_iter = iter(sts_train_dataloader)

    num_steps = max(len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader))

    logger.write("=== Finetuning Interleavingly with Gradient Surgery ===\n")
    for epoch in range(args.epochs):
        model.train()
        # Re-initialize iterators each epoch
        sst_iter = iter(sst_train_dataloader)
        para_iter = iter(para_train_dataloader)
        sts_iter = iter(sts_train_dataloader)

        train_loss = 0.0
        num_batches = 0

        for step in range(num_steps):
            # Fetch one batch per task if available
            tasks = []
            if step < len(sst_train_dataloader):
                tasks.append('sst')
            if step < len(para_train_dataloader):
                tasks.append('para')
            if step < len(sts_train_dataloader):
                tasks.append('sts')

            if not tasks:
                break

            optimizer.zero_grad()

            # We will store gradients separately for each task
            # Strategy:
            # 1. Forward & backward for each task separately
            # 2. Store gradients
            # 3. Apply gradient surgery
            # 4. Finally do an optimizer step

            # Create a dict to store parameter gradients per task
            named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
            grads_per_task = {}

            for task in tasks:
                # Zero out gradients before each backward
                model.zero_grad(set_to_none=True)

                if task == 'sst':
                    try:
                        batch = next(sst_iter)
                    except StopIteration:
                        continue
                    b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                elif task == 'para':
                    try:
                        batch = next(para_iter)
                    except StopIteration:
                        continue
                    b1_ids = batch['token_ids_1'].to(device)
                    b1_mask = batch['attention_mask_1'].to(device)
                    b2_ids = batch['token_ids_2'].to(device)
                    b2_mask = batch['attention_mask_2'].to(device)
                    labels = batch['labels'].float().to(device)
                    logits = model.predict_paraphrase(b1_ids, b1_mask, b2_ids, b2_mask)
                    loss = bce_loss_fn(logits, labels) / args.batch_size
                else:  # sts
                    try:
                        batch = next(sts_iter)
                    except StopIteration:
                        continue
                    b1_ids = batch['token_ids_1'].to(device)
                    b1_mask = batch['attention_mask_1'].to(device)
                    b2_ids = batch['token_ids_2'].to(device)
                    b2_mask = batch['attention_mask_2'].to(device)
                    labels = batch['labels'].float().to(device)
                    logits = model.predict_similarity(b1_ids, b1_mask, b2_ids, b2_mask)
                    loss = mse_loss_fn(logits, labels) / args.batch_size

                loss.backward(retain_graph=True)  # Retain graph in case we need multiple tasks
                # Store gradients
                grads = {}
                for name, p in named_params.items():
                    if p.grad is not None:
                        grads[name] = p.grad.detach().clone()
                    else:
                        grads[name] = torch.zeros_like(p)
                grads_per_task[task] = grads

                train_loss += loss.item()
                num_batches += 1

            # Now apply Gradient Surgery
            # For simplicity, we do pairwise gradient surgery among tasks (if you have 3 tasks).
            # Check for conflicts and fix them:
            def dot_product(g1, g2):
                return sum((g1[name] * g2[name]).sum() for name in g1)

            def norm_square(g):
                return sum((g[name]**2).sum() for name in g)

            # If we have multiple tasks, check conflicts pairwise
            task_list = list(grads_per_task.keys())
            for i in range(len(task_list)):
                for j in range(i+1, len(task_list)):
                    t1 = task_list[i]
                    t2 = task_list[j]
                    g1 = grads_per_task[t1]
                    g2 = grads_per_task[t2]

                    dp = dot_product(g1, g2)
                    if dp < 0:
                        # Project g1 onto the normal plane of g2
                        g2_norm_sq = norm_square(g2)
                        coeff = dp / g2_norm_sq
                        # g1 <- g1 - coeff * g2
                        for name in g1:
                            g1[name] = g1[name] - coeff * g2[name]

                        grads_per_task[t1] = g1

            # Combine all task gradients by summation after fixing conflicts
            # (Alternatively, you might want some weighting)
            final_grads = {name: torch.zeros_like(p) for name, p in named_params.items()}
            for t in grads_per_task:
                for name in final_grads:
                    final_grads[name] += grads_per_task[t][name]

            # Set model gradients to these final combined gradients
            model.zero_grad(set_to_none=True)
            for name, p in named_params.items():
                p.grad = final_grads[name]

            # Update model parameters
            optimizer.step()

        train_loss = train_loss / (num_batches if num_batches > 0 else 1)
        sst_dev_acc, _, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        para_dev_acc, _ = evaluate_paraphrase(para_dev_dataloader, model, device)
        sts_dev_pearson = evaluate_sts(sts_dev_dataloader, model, device)

        improved = False
        if sst_dev_acc > best_sst_acc:
            best_sst_acc = sst_dev_acc
            improved = True
        if para_dev_acc > best_para_acc:
            best_para_acc = para_dev_acc
            improved = True
        if sts_dev_pearson > best_sts_pearson:
            best_sts_pearson = sts_dev_pearson
            improved = True

        if improved:
            save_model(model, optimizer, args, finetune_config, args.finetune_filepath)

        logger.write(f"[Finetune w/ GS] Epoch {epoch}: train_loss={train_loss:.3f}, SST_dev_acc={sst_dev_acc:.3f}, Para_dev_acc={para_dev_acc:.3f}, STS_dev_pearson={sts_dev_pearson:.3f}\n")

    logger.write("Finetuning with Gradient Surgery finished.\n")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        # Load model from args.filepath instead of args.finetune_filepath
        saved = torch.load(args.filepath, map_location=device)
        config = saved['model_config']

        model = MultitaskBERT(config).to(device)
        model.load_state_dict(saved['model'])

        # Set to evaluation mode - no weights will be updated
        model.eval()

        print(f"Loaded model to test from {args.filepath}")

        # Run evaluation on dev/test sets without updating weights
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
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--pretrain_filepath", type=str, default=f'pretrain-{{epochs}}-{{lr}}-multitask-{timestamp}.pt')
    parser.add_argument("--finetune_filepath", type=str, default=f'finetune-{{epochs}}-{{lr}}-multitask-{timestamp}.pt')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--log_file", type=str, default=f'log-{timestamp}.txt')

    parser.add_argument("--option", type=str,
                        help='pretrain: freeze BERT and train heads; finetune: train BERT + heads',
                        choices=('pretrain','finetune'), default="pretrain")

    args = parser.parse_args()
    args.pretrain_filepath = args.pretrain_filepath.format(epochs=args.epochs, lr=args.lr)
    args.finetune_filepath = args.finetune_filepath.format(epochs=args.epochs, lr=args.lr)
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()

        def flush(self):
            self.log.flush()


    logger = Logger(args.log_file)

    if args.option == "pretrain":
        # Perform pretraining
        pretrain_multitask_sequential(args, logger)

        # After pretraining is done, set args.filepath to the pretrain_filepath
        # so that test_model knows which checkpoint to load.
        args.filepath = args.pretrain_filepath
        test_model(args)

    elif args.option == "finetune":
        # Perform finetuning
        finetune_multitask_interleaving(args, logger)

        # After finetuning is done, set args.filepath to the finetune_filepath
        # so that test_model knows which checkpoint to load.
        args.filepath = args.finetune_filepath
        test_model(args)

    logger.write("All done.\n")
    logger.log.close()
