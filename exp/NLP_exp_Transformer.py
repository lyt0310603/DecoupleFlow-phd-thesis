import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from DecoupleFlow import DecoupleFlow
from DecoupleFlow.utils_nlp import get_data, get_word_vector
import numpy as np
import json
import argparse
import time
import math
import random


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer'])
    parser.add_argument('--dataset', type=str, default='ag_news', choices=['ag_news', 'imdb', 'dbpedia_14'])
    parser.add_argument('--batch_size', type=int, default=64, choices=[64, 128, 256, 512, 1024])
    parser.add_argument('--arch', type=str, default='BP', choices=['BP', 'SCPL', 'DeInfo'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--max_len', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42, choices=[41, 42, 43, 44, 45])

    args = parser.parse_args()
    return args

def set_seed(seed):
    """
    設定隨機種子，確保實驗可重複性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_dataloader(dataset, batch_size, max_len):
    args = {}
    args['dataset'] = dataset
    if dataset == 'ag_news':
        args['max_len'] = 60 if max_len is None else int(max_len)
    elif dataset == 'imdb':
        args['max_len'] = 350 if max_len is None else int(max_len)
    elif dataset == 'dbpedia_14':
        args['max_len'] = 400 if max_len is None else int(max_len)
    args['train_bsz'] = batch_size
    args['test_bsz'] = batch_size
    args['noise_rate'] = 0

    train_loader, test_loader, n_classes, vocab = get_data(args)
    word_vec = get_word_vector(vocab)
    
    return train_loader, test_loader, n_classes, vocab, word_vec, args['max_len']

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, x) -> torch.Tensor:
        inputs = x
        
        word_embeddings = self.embedding(inputs) * math.sqrt(self.d_model)
        
        seq_len = word_embeddings.size(1)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
        
        position_embeddings = self.pos_embedding(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        result = self.dropout(embeddings)
        return result, result.mean(dim=1)

def TransformertoFC(x):
    return x.mean(dim=1)

def get_base_model(n_classes, word_vec, vocab_size, max_len):
    model = nn.Sequential(
        PositionalEncoding(vocab_size, 512, max_len=max_len),
        nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        ),
        nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        ),
        torch.nn.Linear(512, 300),
        torch.nn.Tanh(),
        torch.nn.Linear(300, n_classes)
    )
    device_map = {
        "cuda:0": 1,
        "cuda:1": 1,
        "cuda:2": 1,
        "cuda:3": 3,
    }

    return model, device_map

class BP_Transformer(nn.Module):
    def __init__(self, n_classes, word_vec, vocab_size, max_len):
        super(BP_Transformer, self).__init__()
        self.embedding = PositionalEncoding(vocab_size, 512, max_len=max_len)
        self.transformer1 = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer2 = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 300),
            nn.Tanh(),
            nn.Linear(300, n_classes)
        )
    
    def forward(self, x, mask):
        x, _ = self.embedding(x)
        x = self.transformer1(x, src_key_padding_mask=mask)
        x = self.transformer2(x, src_key_padding_mask=mask)
        x = self.fc(x.mean(dim=1))
        return x

def get_model(n_classes, args, word_vec, vocab_size, max_len):
    
    if args.arch == 'BP':
        model = BP_Transformer(n_classes, word_vec, vocab_size, max_len)
        model = model.to('cuda:0')
    else:
        model, device_map = get_base_model(n_classes, word_vec, vocab_size, max_len)
        if args.arch == 'SCPL':
            model = DecoupleFlow(
                model, 
                device_map, 
                loss_fn='CL', 
                projector_type='i',
                optimizer_param={'lr': args.lr},
                transform_funcs=[None, None, None, TransformertoFC]
            )
        elif args.arch == 'DeInfo':
            model = DecoupleFlow(
                model, 
                device_map, 
                loss_fn='DeInfo', 
                projector_type='i',
                num_classes=n_classes,
                optimizer_param={'lr': args.lr},
                transform_funcs=[None, None, None, TransformertoFC],
            )
    return model

def BP_train(model, trainloader, criterion, optimizer):    
    total_loss = 0
    total_time = 0
    for i, (texts, labels, mask) in enumerate(trainloader):        
        texts, labels, mask = texts.to('cuda:0'), labels.to('cuda:0'), mask.to('cuda:0')

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(texts, mask)
        loss = criterion(outputs, labels)        
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss.item()

    return total_loss / len(trainloader), total_time

def BP_eval(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for texts, labels, mask in testloader:
            texts, labels, mask = texts.to('cuda:0'), labels.to('cuda:0'), mask.to('cuda:0')

            outputs = model(texts, mask)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def DecoupleFlow_train(model, trainloader):
    total_loss = 0
    total_time = 0
    for i, (texts, labels, mask) in enumerate(trainloader):

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        outputs, loss, _ = model(texts, labels, mask)

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss

    return total_loss / len(trainloader), total_time

def DecoupleFlow_eval(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for texts, labels, mask in testloader:
            outputs, y_true = model(texts, labels, mask)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)
    return correct / total

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    trainloader, testloader, n_classes, vocab, word_vec, max_len = get_dataloader(
        args.dataset, args.batch_size, args.max_len
    )
    model = get_model(n_classes, args, word_vec, len(vocab), max_len)

    if args.arch == 'BP':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_history = []
    train_time_history = []
    test_acc_history = []

    for epoch in range(args.epochs):
        if args.arch == 'BP':
            train_loss, train_time = BP_train(model, trainloader, criterion, optimizer)
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)
        else:
            train_loss, train_time = DecoupleFlow_train(model, trainloader)
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)

        print(f"epoch [{epoch+1}] train time: {train_time:.2f}, train loss: {train_loss:.4f},", end=' ')

        if args.arch == 'BP':
            eval_acc = BP_eval(model, testloader)
            test_acc_history.append(eval_acc)
        else:
            eval_acc = DecoupleFlow_eval(model, testloader)
            test_acc_history.append(eval_acc)

        print(f"test acc: {eval_acc:.4f}")

    if args.save_path is not None:
        args_dict = vars(args)
        best_acc = np.max(test_acc_history)
        best_epoch = np.argmax(test_acc_history)
        
        result = {
            "args": args_dict,
            "best_acc": float(best_acc),
            "best_epoch": int(best_epoch)
        }
        
        epoch_results = {
            i: {
                "loss": train_loss_history[i],
                "train_time": train_time_history[i],
                "test_acc": test_acc_history[i]
            } for i in range(args.epochs)
        }
        result.update(epoch_results)

        with open(args.save_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {args.save_path}")
