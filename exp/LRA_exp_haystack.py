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
import random
from sklearn.metrics import f1_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM'])
    parser.add_argument('--batch_size', type=int, default=32, choices=[32, 64, 128, 256, 512, 1024])
    parser.add_argument('--arch', type=str, default='BP', choices=['BP', 'SCPL', 'DeInfo'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--max_len', type=str, default=None)

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

def get_dataloader(batch_size, max_len):
    args = {}
    args['dataset'] = 'haystack'
    args['max_len'] = 500 if max_len is None else int(max_len)
    args['train_bsz'] = batch_size
    args['test_bsz'] = batch_size
    args['noise_rate'] = 0

    train_loader, test_loader, n_classes, vocab = get_data(args)
    # word_vec = get_word_vector(vocab)
    
    # return train_loader, test_loader, n_classes, vocab, word_vec, args['max_len']
    return train_loader, test_loader, n_classes, vocab, None, args['max_len']

def LSTMtoLinear(x, h, c):
    return h.mean(dim=0)

def get_base_model(n_classes, word_vec, vocab_size, max_len):
    num_embeddings = len(vocab)
    embedding_dim = 300 # 為了匹配您的 LSTM input_size
    padding_idx = 0
    model = torch.nn.Sequential(
        nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx),
        torch.nn.LSTM(input_size=300, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.Linear(300, 300),
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

class BP_LSTM(nn.Module):
    def __init__(self, n_classes, word_vec, vocab_size, max_len):
        super(BP_LSTM, self).__init__()
        num_embeddings = len(vocab)
        embedding_dim = 300 
        padding_idx = 0
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.lstm1 = nn.LSTM(input_size=300, hidden_size=300, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, n_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, (h, c) = self.lstm2(x)
        x = self.fc(h.mean(dim=0))
        return x

def get_model(n_classes, args, word_vec, vocab_size, max_len):
    
    if args.arch == 'BP':
        model = BP_LSTM(n_classes, word_vec, vocab_size, max_len)
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
                transform_funcs=[None, None, None, LSTMtoLinear]
            )
        elif args.arch == 'DeInfo':
            model = DecoupleFlow(
                model, 
                device_map, 
                loss_fn='DeInfo', 
                projector_type='i',
                num_classes=n_classes,
                optimizer_param={'lr': args.lr},
                transform_funcs=[None, None, None, LSTMtoLinear]
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

        torch.cuda.synchronize(device)

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
            
            outputs = model(texts)
            
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
        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        outputs, loss, _ = model(texts, labels)

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
            
            outputs, y_true = model(texts, labels)
            
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
        args.batch_size,
        args.max_len
    )
    model = get_model(n_classes, args, word_vec, len(vocab), max_len)

    if args.arch == 'BP':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
