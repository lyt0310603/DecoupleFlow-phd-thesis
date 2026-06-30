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
import random


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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM'])
    parser.add_argument('--dataset', type=str, default='imdb', choices=['ag_news', 'imdb', 'dbpedia_14'])
    parser.add_argument('--batch_size', type=int, default=512, choices=[64, 128, 256, 512])
    parser.add_argument('--arch', type=str, default='BP', choices=['BP', 'SCPL', 'DeInfo'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--sim', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=1)

    args = parser.parse_args()
    return args

def get_dataloader(dataset, batch_size):
    args = {}
    args['dataset'] = dataset
    if dataset == 'ag_news':
        args['max_len'] = 60
    elif dataset == 'imdb':
        args['max_len'] = 350
    elif dataset == 'dbpedia_14':
        args['max_len'] = 400
    args['train_bsz'] = batch_size
    args['test_bsz'] = batch_size
    args['noise_rate'] = 0
    
    train_loader, test_loader, n_classes, vocab = get_data(args)
    word_vec = get_word_vector(vocab)
    
    return train_loader, test_loader, n_classes, vocab, word_vec, args['max_len']

def get_base_model(n_classes, word_vec, vocab_size, max_len):
    model = torch.nn.Sequential(
        nn.Embedding.from_pretrained(word_vec, freeze=False),
        torch.nn.LSTM(input_size=300, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True),
        torch.nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True),
    )
    device_map = {
        "cuda:0": 1,
        "cuda:1": 1,
        "cuda:2": 1,
        "cuda:3": 1,
        "cuda:4": 1,
        "cuda:5": 1,
        "cuda:6": 1,
        "cuda:7": 1,        
    }
    extra_classifier = nn.Sequential(
        torch.nn.Tanh(),
        torch.nn.Linear(300, n_classes)
    )

    return model, device_map, extra_classifier

class BP_LSTM(nn.Module):
    def __init__(self, n_classes, word_vec, vocab_size, max_len):
        super(BP_LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vec, freeze=False)
        self.lstm1 = nn.LSTM(input_size=300, hidden_size=300, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True)
        self.lstm5 = nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True)
        self.lstm6 = nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True)
        self.lstm7 = nn.LSTM(input_size=600, hidden_size=300, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, n_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, (h, c) = self.lstm2(x)
        x, (h, c) = self.lstm3(x)
        x, (h, c) = self.lstm4(x)
        x, (h, c) = self.lstm5(x)
        x, (h, c) = self.lstm6(x)
        x, (h, c) = self.lstm7(x)        
        x = self.fc(h.mean(dim=0))
        return x

def get_model(n_classes, args, word_vec, vocab_size, max_len, sim, patience):
    
    if args.arch == 'BP':
        model = BP_LSTM(n_classes, word_vec, vocab_size, max_len)
        model = model.to('cuda:0')
    else:
        model, device_map, extra_classifier = get_base_model(n_classes, word_vec, vocab_size, max_len)
        if args.arch == 'SCPL':
            model = DecoupleFlow(
                model, 
                device_map, 
                loss_fn='CL', 
                projector_type='i',
                optimizer_param={'lr': args.lr},
                is_adaptive=True,
                classifier=extra_classifier,
                patiencethreshold=patience,
                cosinesimthreshold=sim
            )
        elif args.arch == 'DeInfo':
            model = DecoupleFlow(
                model, 
                device_map, 
                loss_fn='DeInfo', 
                projector_type='i',
                num_classes=n_classes,
                optimizer_param={'lr': args.lr},
                is_adaptive=True,
                classifier=extra_classifier,
                patiencethreshold=patience,
                cosinesimthreshold=sim
            )
    return model

def BP_train(model, trainloader, criterion, optimizer):    
    total_loss = 0
    total_time = 0
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

        total_loss += loss.item()

    return total_loss / len(trainloader), total_time

def BP_eval(model, testloader):
    correct = 0
    total = 0
    total_time = 0
    model.eval()
    with torch.no_grad():
        for texts, labels, _ in testloader:
            
            texts, labels = texts.to('cuda:0'), labels.to('cuda:0')            

            outputs = model(texts)

            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total, total_time

def DecoupleFlow_train(model, trainloader):
    total_loss = 0
    total_time = 0
    for i, (texts, labels, _) in enumerate(trainloader):

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        outputs, loss, y_true = model(texts, labels)

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss

    return total_loss / len(trainloader), total_time

def DecoupleFlow_eval(model, testloader):
    correct = 0
    total = 0
    total_time = 0
    idxs = []
    model.eval()
    with torch.no_grad():
        for texts, labels, _ in testloader:

            outputs, idx, y_true = model(texts, labels)
            
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)
            idxs.append(idx)

    return correct / total, total_time, idxs

if __name__ == '__main__':
    set_seed(24)
    args = get_args()
    trainloader, testloader, n_classes, vocab, word_vec, max_len = get_dataloader(args.dataset, args.batch_size)
    model = get_model(n_classes, args, word_vec, len(vocab), max_len, args.sim, args.patience)

    if args.arch == 'BP':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_history = []
    train_time_history = []

    MODEL_PATH = './result/model_weights_3lstm_DeInfo.pth'
    if not os.path.exists(MODEL_PATH) or args.arch=='BP':
        for epoch in range(args.epochs):
            if args.arch == 'BP':
                train_loss, train_time = BP_train(model, trainloader, criterion, optimizer)
                train_loss_history.append(train_loss)
                train_time_history.append(train_time)
            else:
                train_loss, train_time = DecoupleFlow_train(model, trainloader)
                train_loss_history.append(train_loss)
                train_time_history.append(train_time)
    
            print(f"epoch [{epoch+1}] train time: {train_time:.2f}, train loss: {train_loss:.4f}")
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        
    eval_time = 0
    if args.arch == 'BP':
        idxs = None
        s_t = time.time()
        eval_acc, eval_time = BP_eval(model, testloader)
        e_t = time.time()
        eval_time = e_t - s_t
        print(f"test acc: {eval_acc:.4f}, test time: {eval_time:.2f}")
    else:
        model.MovetoSingle('cuda:0')
        adaptive_result = {}
        patiences = [1, 2, 3, 4]
        sims = [0.8, 0.9]
        key_count = 0
        for p in patiences:
            for s in sims:
                model.patiencethreshold = p
                model.cosinesimthreshold = s
                s_t = time.time()
                eval_acc, eval_time, idxs = DecoupleFlow_eval(model, testloader)
                e_t = time.time()
                eval_time = e_t - s_t
                adaptive_result[key_count] = {
                    'patience': int(p),
                    'sim': float(s),
                    'eval_acc': float(eval_acc), 
                    'test_time': float(eval_time), 
                    'idxs': idxs
                }
                key_count += 1
                print(f'[p: {p} s: {s}] test acc: {eval_acc:.4f}, test time: {eval_time:.2f}, idxs: {idxs}')

    if not os.path.exists(MODEL_PATH):
        # 訓練完成後
        print("訓練完成，正在儲存模型...")
        
        # 儲存模型的 "state_dict"
        torch.save(model.state_dict(), MODEL_PATH)
        
        print(f"模型參數已儲存至 {MODEL_PATH}")
    
        if args.save_path is not None:
            args_dict = vars(args)
    
            result = {
                "args": args_dict,
            }
            if args.arch == "DeInfo":
                result.update({"adaptive": adaptive_result})
            else:
                result.update({                
                    "eval_acc": float(eval_acc),
                    "eval_time": float(eval_time)
                })
    
            epoch_results = {
                i: {
                    "loss": train_loss_history[i],
                    "train_time": train_time_history[i],
                } for i in range(args.epochs)
            }
            result.update(epoch_results)
            with open(args.save_path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Results saved to {args.save_path}")
    if os.path.exists(MODEL_PATH) and os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            data = json.load(f)

        data["adaptive"] = adaptive_result

        with open(args.save_path, "w") as f:
            json.dump(data, f, indent=4)
            
