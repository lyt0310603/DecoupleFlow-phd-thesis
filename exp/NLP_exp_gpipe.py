import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchgpipe import GPipe
from DecoupleFlow.utils_nlp import get_data, get_word_vector
import numpy as np
import json
import argparse
import time
import random


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM'])
    parser.add_argument('--dataset', type=str, default='ag_news', choices=['ag_news', 'imdb', 'dbpedia_14'])
    parser.add_argument('--batch_size', type=int, default=64, choices=[64, 128, 256, 512, 1024])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--chunks', type=int, default=4, help='GPipe 的 micro-batch 數量')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--max_len', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42, choices=[41, 42, 43, 44, 45, 46, 47, 48, 49, 50])

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

# === GPipe 用的 stage wrapper ===
# GPipe 要求模型是 nn.Sequential，且每個 stage 的輸入/輸出都是 Tensor，
# 因此把 LSTM 模型的四個部分包成各自輸出 Tensor 的子模組，再交給 torchgpipe 切分。
# 所有中間張量的第 0 維都維持為 batch，符合 GPipe 沿 batch 維切 micro-batch 的需求。
class EmbeddingStage(nn.Module):
    def __init__(self, word_vec):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vec, freeze=False)

    def forward(self, x):
        return self.embedding(x)

class LSTMStage(nn.Module):
    def __init__(self, input_size, hidden_size, return_hidden=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.return_hidden = return_hidden

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        if self.return_hidden:
            return h.mean(dim=0)
        return out

class ClassifierStage(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, n_classes)
        )

    def forward(self, x):
        return self.fc(x)

def get_gpipe_model(n_classes, word_vec, chunks=4):
    n_gpus = torch.cuda.device_count()
    if n_gpus < 4:
        raise RuntimeError(f"GPipe 設定需要 4 個 GPU，但目前只偵測到 {n_gpus} 個")

    module = nn.Sequential(
        EmbeddingStage(word_vec),
        LSTMStage(input_size=300, hidden_size=300),
        LSTMStage(input_size=600, hidden_size=300, return_hidden=True),
        ClassifierStage(n_classes),
    )

    model = GPipe(
        module,
        balance=[1, 1, 1, 1],
        devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        chunks=chunks,
        checkpoint='never',
    )
    return model

def GPipe_train(model, trainloader, criterion, optimizer):
    in_device = model.devices[0]
    out_device = model.devices[-1]

    total_loss = 0
    total_time = 0
    for i, (texts, labels, _) in enumerate(trainloader):
        texts = texts.to(in_device)
        labels = labels.to(out_device)

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss.item()

    return total_loss / len(trainloader), total_time

def GPipe_eval(model, testloader):
    in_device = model.devices[0]
    out_device = model.devices[-1]

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for texts, labels, _ in testloader:
            texts = texts.to(in_device)
            labels = labels.to(out_device)

            outputs = model(texts)

            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    trainloader, testloader, n_classes, vocab, word_vec, max_len = get_dataloader(
        args.dataset, args.batch_size, args.max_len
    )
    model = get_gpipe_model(n_classes, word_vec, chunks=args.chunks)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_history = []
    train_time_history = []
    test_acc_history = []

    for epoch in range(args.epochs):
        train_loss, train_time = GPipe_train(model, trainloader, criterion, optimizer)
        train_loss_history.append(train_loss)
        train_time_history.append(train_time)

        print(f"epoch [{epoch+1}] train time: {train_time:.2f}, train loss: {train_loss:.4f},", end=' ')

        eval_acc = GPipe_eval(model, testloader)
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
