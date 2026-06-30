"""Modified from https://github.com/ChengKai-Wang/Supervised-Contrastive-Parallel-Learning"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
import os
from tqdm import tqdm

class CustomDataset():
  def __init__(self, X, Y, num=None):
    self.x = X
    self.y = Y
    self.n_samples = num

  # working for indexing
  def __getitem__(self, index):
    return self.x[index], self.y[index]

  # return the length of our dataset
  def __len__(self):
    if self.n_samples is None:
        return len(self.x)
    return self.n_samples


def my_collate(batch):
    x = torch.cat([torch.stack(a) for a, b in batch])
    y = torch.cat([torch.tensor(b) for a, b in batch])
    return x, y


def set_loader(dataset, train_bsz, test_bsz, augmentation_type):
    if dataset == "cifar10":
        n_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == "cifar100":
        n_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "tinyImageNet":
        n_classes = 200
    elif dataset == "ImageNet":
        n_classes = 1000
    else:
        raise ValueError("Dataset not supported: {}".format(dataset))
    
    if dataset == "cifar10" or dataset == "cifar100":
        normalize = transforms.Normalize(mean=mean, std=std)
        weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if dataset == "tinyImageNet":
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])
        
        weak_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
    if dataset == "ImageNet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        weak_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])

    if augmentation_type == "basic":
        source_transform = weak_transform
        target_transform = None
    elif augmentation_type == "strong":
        source_transform = TwoCropTransform(weak_transform, strong_transform)
        target_transform = TwoCropTransform(None)
    else:
        raise ValueError("Augmentation type not supported: {}".format(augmentation_type))

    
    if dataset == "cifar10":
        train_set = datasets.CIFAR10(root='./cifar10', transform=source_transform, target_transform = target_transform,  download=True)
        test_set = datasets.CIFAR10(root='./cifar10', train=False, transform=test_transform)
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(root='./cifar100', transform=source_transform, target_transform = target_transform, download=True)
        test_set = datasets.CIFAR100(root='./cifar100', train=False, transform=test_transform)
    elif dataset == "tinyImageNet":
        train_set = datasets.ImageFolder('./tiny-imagenet-200/train', transform=source_transform)
        test_set = datasets.ImageFolder('./tiny-imagenet-200/val', transform=test_transform)
    elif dataset == "ImageNet":
        train_set = datasets.ImageFolder('/work/lyt0310603/ImageNet/train', transform=source_transform)
        test_set = datasets.ImageFolder('/work/lyt0310603/ImageNet/val', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bsz, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bsz, shuffle=False, pin_memory=True)

    return train_loader, test_loader, n_classes

# for convolutional neural networks
def conv_layer_bn(in_channels: int, out_channels: int, activation: nn.Module, stride: int=1, bias: bool=False) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, bias = bias, padding = 1)
    bn = nn.BatchNorm2d(out_channels)
    if activation == None:
        return nn.Sequential(conv, bn)
    return nn.Sequential(conv, bn, activation)

def conv_1x1_bn(in_channels: int, out_channels: int, activation: nn.Module = None, stride: int=1, bias: bool=False) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = bias)
    bn = nn.BatchNorm2d(out_channels)
    if activation == None:
        return nn.Sequential(conv, bn)
    return nn.Sequential(conv, bn, activation)

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def add_noise_cifar(loader, noise_rate):
    """ Referenced from https://github.com/PaulAlbert31/LabelNoiseCorrection """
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_rate)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(set(range(10)))
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels

    return noisy_labels

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        if transform1 != None and transform2 == None:
            self.transform2 = transform1
        else:
            self.transform2 = transform2

    def __call__(self, x):
        if self.transform1 == None:
            return [x, x]
        return [self.transform1(x), self.transform2(x)]
