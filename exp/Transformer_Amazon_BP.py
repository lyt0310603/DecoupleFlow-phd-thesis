import argparse
import gc
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import T5Tokenizer

DEFAULT_DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    'data',
    'amazon_review',
)


def get_args():
    parser = argparse.ArgumentParser(description='Amazon Review Transformer BP baseline (single GPU).')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.environ.get('AMAZON_REVIEW_OUTPUT_DIR', DEFAULT_DATA_DIR),
        help='Directory containing preprocessed Parquet files.',
    )
    parser.add_argument(
        '--category',
        type=str,
        default='raw_review_Books',
        help='Parquet filename without extension, e.g. raw_review_Books.',
    )
    parser.add_argument('--val_split_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Optional directory to save epoch checkpoints.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class AmazonReviewDataset(Dataset):
    def __init__(self, hf_dataset_slice):
        self.dataset = hf_dataset_slice

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]

            input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
            original_rating = item['label']

            if (
                original_rating is None
                or not isinstance(original_rating, (int, float))
                or not (1 <= original_rating <= 5)
            ):
                return None

            label = torch.tensor(int(original_rating) - 1, dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label,
            }
        except Exception as e:
            print(f"Warning: skipping sample at index {idx}. Error: {e}")
            return None


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return {}
    return default_collate(batch)


def get_dataloader(data_dir, category, val_split_ratio, seed, batch_size, num_workers):
    selected_parquet_path = os.path.join(data_dir, f"{category}.parquet")

    print(f"Loading Amazon Reviews data: {selected_parquet_path}")
    try:
        full_hf_dataset = HFDataset.from_parquet(selected_parquet_path, cache_dir=None)
        print(f"Loaded {len(full_hf_dataset)} records.")
    except Exception as e:
        raise RuntimeError(f"Failed to load Parquet data: {e}") from e

    shuffled_dataset = full_hf_dataset.shuffle(seed=seed)
    total_size = len(shuffled_dataset)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size

    train_hf_dataset = shuffled_dataset.select(range(train_size))
    eval_hf_dataset = shuffled_dataset.select(range(train_size, total_size))

    train_dataset = AmazonReviewDataset(train_hf_dataset)
    eval_dataset = AmazonReviewDataset(eval_hf_dataset)

    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        generator=g,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )

    print(f"Train DataLoader steps: {len(train_loader)}")
    print(f"Eval DataLoader steps: {len(eval_loader)}")
    return train_loader, eval_loader, 5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings = self.pos_embedding(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, nhead, dim_feedforward, max_len=128):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        self.encoder1 = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder2 = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder3 = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, 300)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(300, num_classes)
        self.d_model = embedding_dim

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        padding_mask = (attention_mask == 0)

        x = self.encoder1(embedded, src_key_padding_mask=padding_mask)
        x = self.encoder2(x, src_key_padding_mask=padding_mask)
        x = self.encoder3(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=1)
        x = self.tanh(self.linear1(x))
        return self.classifier(x)


def get_model(n_classes, vocab_size, device):
    return TransformerEncoderClassifier(
        vocab_size=vocab_size,
        embedding_dim=512,
        hidden_size=512,
        num_classes=n_classes,
        nhead=8,
        dim_feedforward=2048,
    ).to(device)


def BP_train(model, trainloader, criterion, optimizer, device, grad_clip):
    total_loss = 0.0
    total_time = 0.0
    num_batches = 0

    model.train()
    for batch in trainloader:
        if not batch or batch['input_ids'].shape[0] == 0:
            continue

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        torch.cuda.synchronize()
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        torch.cuda.synchronize()
        total_time += time.time() - start_time
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1), total_time


def BP_eval(model, testloader, device):
    correct = 0
    total = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for batch in testloader:
            if not batch or batch['input_ids'].shape[0] == 0:
                continue

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            torch.cuda.synchronize()
            start_time = time.time()

            outputs = model(input_ids, attention_mask)

            torch.cuda.synchronize()
            total_time += time.time() - start_time

            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = (correct / total * 100) if total > 0 else 0.0
    return total_time, acc


def main():
    args = get_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    device = torch.device('cuda:0')

    train_loader, test_loader, n_classes = get_dataloader(
        args.data_dir,
        args.category,
        args.val_split_ratio,
        args.seed,
        args.batch_size,
        args.num_workers,
    )

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = get_model(n_classes, tokenizer.vocab_size, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    print(model)

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loss_history = []
    train_time_history = []
    test_time_history = []
    test_acc_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_time = BP_train(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )
        test_time, test_acc = BP_eval(model, test_loader, device)

        train_loss_history.append(train_loss)
        train_time_history.append(train_time)
        test_time_history.append(test_time)
        test_acc_history.append(test_acc)

        print(
            f"epoch [{epoch}/{args.epochs}] "
            f"train time: {train_time:.2f}, loss: {train_loss:.4f}, "
            f"test time: {test_time:.2f}, test acc: {test_acc:.2f}%"
        )

        if args.checkpoint_dir is not None:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"Transformer_BP_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        gc.collect()

    if args.save_path is not None:
        best_acc = float(np.max(test_acc_history))
        best_epoch = int(np.argmax(test_acc_history))
        result = {
            "args": vars(args),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
        }
        result.update({
            str(i + 1): {
                "train_loss": train_loss_history[i],
                "train_time": train_time_history[i],
                "test_time": test_time_history[i],
                "test_acc": test_acc_history[i],
            }
            for i in range(args.epochs)
        })
        with open(args.save_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
