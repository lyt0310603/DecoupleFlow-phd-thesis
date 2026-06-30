import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchgpipe import GPipe
from DecoupleFlow.utils_vision import set_loader
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

    parser.add_argument('--model', type=str, default='VGG', choices=['VGG'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyImageNet'])
    parser.add_argument('--batch_size', type=int, default=64, choices=[32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42, choices=[41, 42, 43, 44, 45])
    parser.add_argument('--chunks', type=int, default=4, help='GPipe 的 micro-batch 數量')

    args = parser.parse_args()
    return args

def get_dataloader(dataset, batch_size):
    trainloader, testloader, n_classes = set_loader(dataset, batch_size, batch_size, "strong")
    return trainloader, testloader, n_classes


class Vision_VGG(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone0 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.backbone1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.backbone2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(2500),
            nn.ReLU(),
            nn.Linear(2500, n_classes),
        )

    def forward(self, x):
        x = self.backbone0(x)
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.classifier(x)
        return x


def get_gpipe_vgg_model(n_classes, dataset, chunks=4):
    """
    用 torchgpipe 把 VGG 切成四段（backbone0 / backbone1 / backbone2 / classifier），
    分別放到 cuda:0~cuda:3。
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus < 4:
        raise RuntimeError(f"GPipe 設定需要 4 個 GPU，但目前只偵測到 {n_gpus} 個")

    base = Vision_VGG(n_classes)
    module = nn.Sequential(
        base.backbone0,
        base.backbone1,
        base.backbone2,
        base.classifier,
    )

    img_size = 64 if dataset == 'tinyImageNet' else 32
    module.eval()
    with torch.no_grad():
        module(torch.randn(2, 3, img_size, img_size))

    model = GPipe(
        module,
        balance=[1, 1, 1, 1],
        devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        chunks=chunks,
        checkpoint='never',
        deferred_batch_norm=True,
    )
    return model


def GPipe_train(model, trainloader, criterion, optimizer, dataset, lr_scheduler):
    in_device = model.devices[0]
    out_device = model.devices[-1]

    total_loss = 0
    total_time = 0
    for i, (images, labels) in enumerate(trainloader):
        if dataset == 'cifar10' or dataset == 'cifar100':
            images = torch.cat(images)
            labels = torch.cat(labels)
        elif dataset == 'tinyImageNet':
            images = torch.cat(images)
            labels = torch.cat([labels, labels])

        images = images.to(in_device)
        labels = labels.to(out_device)

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


def GPipe_eval(model, testloader, dataset):
    in_device = model.devices[0]
    out_device = model.devices[-1]

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(in_device)
            labels = labels.to(out_device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    trainloader, testloader, n_classes = get_dataloader(args.dataset, args.batch_size)
    model = get_gpipe_vgg_model(n_classes, args.dataset, chunks=args.chunks)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

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
        elif args.arch == 'GPipe':
            train_loss, train_time, lr = GPipe_train(model, trainloader, criterion, optimizer, args.dataset, lr_scheduler)
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)
        else:
            train_loss, train_time, lr = RegSCPL_train(model, trainloader, args.dataset)
            train_loss_history.append(train_loss)
            train_time_history.append(train_time)

        print(
            f"epoch [{epoch+1}] train time: {train_time:.2f}, "
            f"train loss: {train_loss:.4f}, lr: {lr:.6f},",
            end=' '
        )

        eval_acc = GPipe_eval(model, testloader, args.dataset)
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
