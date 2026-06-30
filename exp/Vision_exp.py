import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from DecoupleFlow import DecoupleFlow
from DecoupleFlow.utils import LARS
from DecoupleFlow.utils_vision import set_loader, conv_layer_bn, conv_1x1_bn
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

    parser.add_argument('--model', type=str, default='VGG', choices=['VGG', 'ResNet'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyImageNet'])
    parser.add_argument('--batch_size', type=int, default=64, choices=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument('--arch', type=str, default='BP', choices=['BP', 'SCPL', 'DeInfo'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42, choices=[41, 42, 43, 44, 45])

    args = parser.parse_args()
    return args

def get_dataloader(dataset, batch_size):
    trainloader, testloader, n_classes = set_loader(dataset, batch_size, batch_size, "strong")
    return trainloader, testloader, n_classes

class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer_bn(in_channels, out_channels, nn.LeakyReLU(inplace=True), stride, False)
        self.conv2 = conv_layer_bn(out_channels, out_channels, None, 1, False)
        self.relu = nn.LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        if stride != 1:
            # self.shortcut = conv_layer_bn(in_channels, out_channels, None, stride, False) # Original SCPL settings
            self.shortcut = conv_1x1_bn(in_channels, out_channels, None, stride, False) # New settings. Maybe this is the correct setting for ResNet18

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + self.shortcut(x))
        return out

def get_base_model(model, n_classes, arch):
    if model == 'VGG':
        model = nn.Sequential(
            # Backbone-0
            nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        
            # Backbone-1
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        
            # Backbone-2
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            # Predictor
            nn.Flatten(),
            nn.LazyLinear(2500),
            nn.ReLU(),
            nn.Linear(2500, n_classes)
        )
        device_map = {
            "cuda:0": 7,
            "cuda:1": 7,
            "cuda:2": 4,
            "cuda:3": 4,
        }
    elif model == 'ResNet' and arch == 'DeInfo':
        model = nn.Sequential(
            # Head
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            # Backbone-0 (64 -> 64)
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            
            # Backbone-1 (64 -> 128)
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            
            
            # Backbone-2 (128 -> 256)
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),  
            BasicBlock(256, 256),
            
            # Predictor            
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        device_map = {
            "cuda:0": 5,
            "cuda:1": 3,
            "cuda:2": 3,   # DeInfo 下拿掉 AdaptiveAvgPool2d
            "cuda:3": 4,   # DeInfo 改為多層 mlp
        }
    else:
        model = nn.Sequential(
            # Head
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            # Backbone-0 (64 -> 64)
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            
            # Backbone-1 (64 -> 128)
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            
            
            # Backbone-2 (128 -> 256)
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),  
            BasicBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)), 
            
            # Predictor            
            nn.Flatten(),
            nn.Linear(256, n_classes)
        )
        device_map = {
            "cuda:0": 5,
            "cuda:1": 3,
            "cuda:2": 4,   
            "cuda:3": 2,   
        }

    return model, device_map

def get_model(model, n_classes, args):
    model, device_map = get_base_model(model, n_classes, args.arch)
    if args.arch == 'BP':
        model = model.to('cuda:0')
    elif args.arch == 'SCPL':
        model = DecoupleFlow(
            model, 
            device_map, 
            loss_fn='CL', 
            projector_type='mlp',
            optimizer_param={'lr': args.lr},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_param={'T_max': args.epochs, 'eta_min': args.lr*0.01}
        )
    elif args.arch == 'DeInfo':
        model = DecoupleFlow(
            model, 
            device_map, 
            loss_fn='DeInfo', 
            projector_type='mlp',
            num_classes=n_classes,
            optimizer_fn="LARS",
            optimizer_param= {
                "lr":args.lr,
                "weight_decay_filter":LARS.exclude_bias_and_norm,
                "lars_adaptation_filter":LARS.exclude_bias_and_norm,
                "weight_decay":1e-4,
            },
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_param={'T_max': args.epochs, 'eta_min': args.lr*0.01}
        )
    return model

def BP_train(model, trainloader, criterion, optimizer, dataset, lr_scheduler):    
    total_loss = 0
    total_time = 0
    for i, (images, labels) in enumerate(trainloader):
        if dataset == 'cifar10' or dataset == 'cifar100':
            images = torch.cat(images)
            labels = torch.cat(labels)
        elif dataset == 'tinyImageNet':
            images = torch.cat(images)
            labels = torch.cat([labels, labels])
        
        images, labels = images.to('cuda:0'), labels.to('cuda:0')

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)        
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss.item()
    lr_scheduler.step()

    return total_loss / len(trainloader), total_time, lr_scheduler.get_last_lr()[0]

def BP_eval(model, testloader, dataset):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to('cuda:0'), labels.to('cuda:0')

            outputs = model(images)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def DecoupleFlow_train(model, trainloader, dataset):
    total_loss = 0
    total_time = 0
    for i, (images, labels) in enumerate(trainloader):
        if dataset == 'cifar10' or dataset == 'cifar100':
            images = torch.cat(images)
            labels = torch.cat(labels)
        elif dataset == 'tinyImageNet':
            images = torch.cat(images)
            labels = torch.cat([labels, labels])

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        outputs, loss, _ = model(images, labels)

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss
    model.scheduler_step()

    return total_loss / len(trainloader), total_time, model.get_lr()

def DecoupleFlow_eval(model, testloader, dataset):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs, y_true = model(images, labels)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)
    return correct / total

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    trainloader, testloader, n_classes = get_dataloader(args.dataset, args.batch_size)
    model = get_model(args.model, n_classes, args)

    if args.arch == 'BP':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    train_loss_history = []
    train_time_history = []
    test_acc_history = []

    for epoch in range(args.epochs):
        if args.arch == 'BP':
            s_t = time.time()
            train_loss, train_time, lr = BP_train(model, trainloader, criterion, optimizer, args.dataset, lr_scheduler)
            e_t = time.time()
            train_time = e_t-s_t
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)
        else:
            train_loss, train_time, lr = DecoupleFlow_train(model, trainloader, args.dataset)
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)

        print(f"epoch [{epoch+1}] train time: {train_time:.2f}, train loss: {train_loss:.4f}, lr: {lr:.6f},", end=' ')

        if args.arch == 'BP':
            eval_acc = BP_eval(model, testloader, args.dataset)
            test_acc_history.append(eval_acc)
        else:
            eval_acc = DecoupleFlow_eval(model, testloader, args.dataset)
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
