#!/usr/bin/env python3

import argparse
import torch
from types import SimpleNamespace

# Import your modules
from multitask_classifier import MultitaskBERT, test_model, seed_everything
# Note: Adjust imports as needed. For example, if your code is structured differently,
# make sure to import from the correct files.
# The imports here assume that 'multitask_classifier.py' is the file where the model and
# test_model function are defined.

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--option", type=str,
                        help='pretrain: freeze BERT; finetune: update BERT',
                        choices=('pretrain', 'finetune'), default="pretrain")

    # Paths to output predictions
    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11711)

    # The path to the previously saved model checkpoint
    # Update this to match the filename you used when training.
    parser.add_argument("--filepath", type=str, default="pretrain-10-0.001-multitask.pt")

    args = parser.parse_args()
    seed_everything(args.seed)

    # Run the test_model function which:
    # 1. Loads the saved model from args.filepath
    # 2. Evaluates on dev and test sets, printing out metrics
    # 3. Writes predictions to the specified output CSV files
    test_model(args)
