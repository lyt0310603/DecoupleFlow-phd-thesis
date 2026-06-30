import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from DecoupleFlow import DecoupleFlow
from DecoupleFlow.utils_nlp import get_data
import numpy as np
import json
import argparse
import time
import math
import random
from sklearn.metrics import f1_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer'])
    parser.add_argument('--batch_size', type=int, default=512, choices=[32, 64, 128, 256, 512, 1024])
    parser.add_argument('--arch', type=str, default='BP', choices=['BP', 'SCPL', 'DeInfo'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--max_len', type=str, default=None)
    parser.add_argument('--train_classes', type=int, default=3, choices=[2, 3])
    parser.add_argument('--test_classes', type=int, default=3, choices=[2, 3])

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

def bytes_to_gb(bytes_val):
    return bytes_val / (1024**3)

def get_dataloader(batch_size, max_len, train_classes, test_classes):
    args = {
        'dataset': 'contractnli',
        'max_len': 512 if max_len is None else int(max_len),
        'train_bsz': batch_size,
        'test_bsz': 32,
        'noise_rate': 0,
        'train_classes': train_classes,
        'test_classes': test_classes,
    }

    train_loader, test_loader, n_classes, vocab = get_data(args)
    return train_loader, test_loader, n_classes, vocab, None, args['max_len']

def seq_mean_pool(x, mask=None):
    if mask is not None:
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        return x.sum(dim=1) / lengths
    return x.mean(dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.d_model = d_model

        # 固定 (非學習) sinusoidal 位置編碼：跨不同 max_len 數值穩定，
        # 長位置不需靠資料訓練，避免大 max_len 時後段位置學不動。
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)

    def forward(self, x, mask=None) -> torch.Tensor:
        inputs = x

        word_embeddings = self.embedding(inputs) * math.sqrt(self.d_model)

        seq_len = word_embeddings.size(1)

        position_embeddings = self.pos_encoding[:seq_len].unsqueeze(0)

        embeddings = word_embeddings + position_embeddings
        result = self.dropout(embeddings)
        return result, seq_mean_pool(result, mask)

def get_base_model(n_classes, word_vec, vocab_size, max_len):
    model = nn.Sequential(
        PositionalEncoding(vocab_size, 512, max_len=max_len),
        nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True),
        nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True),
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
        x, _ = self.embedding(x, mask)
        x = self.transformer1(x, src_key_padding_mask=mask)
        x = self.transformer2(x, src_key_padding_mask=mask)
        x = seq_mean_pool(x, mask)
        x = self.fc(x)
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
                projector_type='mlp',
                optimizer_fn=optim.AdamW,
                optimizer_param={'lr': args.lr},
                transform_funcs=[None, None, None, seq_mean_pool],
            )
        elif args.arch == 'DeInfo':
            model = DecoupleFlow(
                model, 
                device_map, 
                loss_fn='DeInfo', 
                projector_type='mlp',
                num_classes=n_classes,
                optimizer_fn=optim.AdamW,
                optimizer_param={'lr': args.lr},
                transform_funcs=[None, None, None, seq_mean_pool],
            )
    return model

def BP_train(model, trainloader, criterion, optimizer):    
    total_loss = 0
    total_time = 0

    device = 'cuda:0'
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    
    for i, (texts, labels, _) in enumerate(trainloader):        
        texts, labels = texts.to('cuda:0'), labels.to('cuda:0')
        mask = (texts == 0)

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
        
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_gb = bytes_to_gb(peak_mem_bytes)

    return total_loss / len(trainloader), total_time, peak_mem_gb

def BP_eval(model, testloader):
    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for texts, labels, _ in testloader:
            texts, labels = texts.to('cuda:0'), labels.to('cuda:0')
            mask = (texts == 0)

            outputs = model(texts, mask)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true_all.extend(labels.detach().cpu().tolist())
            y_pred_all.extend(preds.detach().cpu().tolist())
    acc = correct / total
    macro_f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    return acc, macro_f1

def DecoupleFlow_train(model, trainloader):
    total_loss = 0
    total_time = 0

    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    for dev in devices:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    
    for i, (texts, labels, _) in enumerate(trainloader):
        mask = (texts == 0)

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        outputs, loss, _ = model(texts, labels, mask)

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss

    for dev in devices:
        torch.cuda.synchronize(dev)
    
    peak_memory_all_gpus = {}
    for dev in devices:
        peak_mem_bytes = torch.cuda.max_memory_allocated(dev)
        peak_memory_all_gpus[str(dev)] = bytes_to_gb(peak_mem_bytes)

    return total_loss / len(trainloader), total_time, peak_memory_all_gpus

def DecoupleFlow_eval(model, testloader):
    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for texts, labels, _ in testloader:
            mask = (texts == 0)
            outputs, y_true = model(texts, labels, mask)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)
            y_true_all.extend(y_true.detach().cpu().tolist())
            y_pred_all.extend(preds.detach().cpu().tolist())
    acc = correct / total
    macro_f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    return acc, macro_f1

if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    trainloader, testloader, n_classes, vocab, word_vec, max_len = get_dataloader(
        args.batch_size, args.max_len, args.train_classes, args.test_classes
    )
    model = get_model(n_classes, args, word_vec, len(vocab), max_len)

    if args.arch == 'BP':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_loss_history = []
    train_time_history = []
    mem_history = []
    test_acc_history = []
    test_macro_f1_history = []

    for epoch in range(args.epochs):
        if args.arch == 'BP':
            train_loss, train_time, peak_mem_gb = BP_train(model, trainloader, criterion, optimizer)
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)
            mem_history.append(peak_mem_gb)
        else:
            train_loss, train_time, peak_memory_all_gpus = DecoupleFlow_train(model, trainloader)
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)
            mem_history.append(peak_memory_all_gpus)

        print(f"epoch [{epoch+1}] train time: {train_time:.2f}, train loss: {train_loss:.4f},", end=' ')

        if args.arch == 'BP':
            eval_acc, eval_macro_f1 = BP_eval(model, testloader)
            test_acc_history.append(eval_acc)
            test_macro_f1_history.append(eval_macro_f1)
        else:
            eval_acc, eval_macro_f1 = DecoupleFlow_eval(model, testloader)
            test_acc_history.append(eval_acc)
            test_macro_f1_history.append(eval_macro_f1)

        print(f"test acc: {eval_acc:.4f}, test macro-f1: {eval_macro_f1:.4f}")

    if args.save_path is not None:
        args_dict = vars(args)
        best_acc = np.max(test_acc_history)
        best_epoch = np.argmax(test_acc_history)
        best_macro_f1 = np.max(test_macro_f1_history)
        
        result = {
            "args": args_dict,
            "best_acc": float(best_acc),
            "best_epoch": int(best_epoch),
            "best_macro_f1": float(best_macro_f1)
        }
        
        epoch_results = {
            i: {
                "loss": train_loss_history[i],
                "train_time": train_time_history[i],
                "test_acc": test_acc_history[i],
                "test_macro_f1": test_macro_f1_history[i],
                "peak_mem": mem_history[i]
            } for i in range(args.epochs)
        }
        result.update(epoch_results)

        with open(args.save_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {args.save_path}")
