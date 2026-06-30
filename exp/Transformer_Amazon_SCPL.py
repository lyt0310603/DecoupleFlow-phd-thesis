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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DecoupleFlow import DecoupleFlow

DEFAULT_DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    'data',
    'amazon_review',
)


def get_args():
    parser = argparse.ArgumentParser(description='Amazon Review Transformer with DecoupleFlow (SCPL).')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.environ.get('AMAZON_REVIEW_OUTPUT_DIR', DEFAULT_DATA_DIR),
        help='Directory containing preprocessed Parquet files.',
    )
    parser.add_argument(
        '--category',
        type=str,
        default='raw_review_Subscription_Boxes',
        help='Parquet filename without extension, e.g. raw_review_Books.',
    )
    parser.add_argument('--val_split_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
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
            print(f"警告: 讀取或處理索引 {idx} 時發生異常，忽略該樣本。錯誤: {e}")
            return None


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return {}
    return default_collate(batch)


def get_dataloader(data_dir, category, val_split_ratio, seed, batch_size, num_workers):
    selected_parquet_path = os.path.join(data_dir, f"{category}.parquet")

    print(f"載入 Amazon Reviews 數據: {selected_parquet_path}")
    try:
        full_hf_dataset = HFDataset.from_parquet(selected_parquet_path, cache_dir=None)
        print(f"成功載入數據，總計 {len(full_hf_dataset)} 條記錄。")
    except Exception as e:
        raise RuntimeError(f"從 Parquet 載入數據失敗: {e}") from e

    shuffled_dataset = full_hf_dataset.shuffle(seed=seed)
    total_size = len(shuffled_dataset)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size

    train_hf_dataset = shuffled_dataset.select(range(train_size))
    eval_hf_dataset = shuffled_dataset.select(range(train_size, total_size))

    train_dataset = AmazonReviewDataset(train_hf_dataset)
    eval_dataset = AmazonReviewDataset(eval_hf_dataset)

    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(eval_dataset)}")

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

    print(f"訓練 DataLoader steps: {len(train_loader)}")
    print(f"驗證 DataLoader steps: {len(eval_loader)}")
    return train_loader, eval_loader, 5


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, x) -> torch.Tensor:
        word_embeddings = self.embedding(x) * math.sqrt(self.d_model)
        seq_len = word_embeddings.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings = self.pos_embedding(position_ids)
        embeddings = word_embeddings + position_embeddings
        result = self.dropout(embeddings)
        return result, result


def transform_to_fc(x):
    return x.mean(dim=1)


def get_base_model(n_classes, vocab_size, max_len=128):
    embed_size = 512
    hidden_size = 512
    nhead = 8
    dim_feedforward = 2048

    model = nn.Sequential(
        PositionalEncoding(vocab_size, embed_size, max_len=max_len),
        nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        ),
        nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        ),
        nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        ),
        nn.Linear(hidden_size, 300),
        nn.Tanh(),
        nn.Linear(300, n_classes),
    )
    device_map = {
        "cuda:0": 1,
        "cuda:1": 1,
        "cuda:2": 1,
        "cuda:3": 4,
    }
    return model, device_map


def get_model(n_classes, vocab_size, lr):
    model, device_map = get_base_model(n_classes, vocab_size)
    model = DecoupleFlow(
        model,
        device_map,
        loss_fn='CL',
        projector_type='mlp',
        optimizer_fn=optim.AdamW,
        optimizer_param={'lr': lr},
        transform_funcs=[None, None, None, transform_to_fc],
    )
    return model


def DecoupleFlow_train(model, trainloader):
    total_loss = 0.0
    total_time = 0.0

    for batch in trainloader:
        if not batch or batch['input_ids'].shape[0] == 0:
            continue

        input_ids = batch['input_ids']
        labels = batch['labels']
        mask = (batch['attention_mask'] == 0)

        model.train()

        torch.cuda.synchronize()
        start_time = time.time()

        _, loss, _ = model(input_ids, labels, mask)

        torch.cuda.synchronize()
        total_time += time.time() - start_time
        total_loss += loss

    return total_loss / len(trainloader), total_time


def DecoupleFlow_eval(model, testloader):
    correct = 0
    total = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for batch in testloader:
            if not batch or batch['input_ids'].shape[0] == 0:
                continue

            input_ids = batch['input_ids']
            labels = batch['labels']
            mask = (batch['attention_mask'] == 0)

            torch.cuda.synchronize()
            start_time = time.time()

            outputs, y_true = model(input_ids, labels, mask)

            torch.cuda.synchronize()
            total_time += time.time() - start_time

            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)

    acc = (correct / total * 100) if total > 0 else 0.0
    return total_time, acc


def main():
    args = get_args()
    set_seed(args.seed)

    train_loader, test_loader, n_classes = get_dataloader(
        args.data_dir,
        args.category,
        args.val_split_ratio,
        args.seed,
        args.batch_size,
        args.num_workers,
    )

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = get_model(n_classes, tokenizer.vocab_size, args.lr)
    print(model)

    train_loss_history = []
    train_time_history = []
    test_time_history = []
    test_acc_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_time = DecoupleFlow_train(model, train_loader)
        test_time, test_acc = DecoupleFlow_eval(model, test_loader)

        train_loss_history.append(train_loss)
        train_time_history.append(train_time)
        test_time_history.append(test_time)
        test_acc_history.append(test_acc)

        print(
            f"epoch [{epoch}/{args.epochs}] "
            f"train time: {train_time:.2f}, loss: {train_loss:.4f}, "
            f"test time: {test_time:.2f}, test acc: {test_acc:.2f}%"
        )

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
