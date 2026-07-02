# DecoupleFlow-thesis

Implementation and experiment scripts for **DecoupleFlow**, a decoupled multi-GPU training framework developed for thesis research.

DecoupleFlow splits a deep model into blocks across GPUs and trains each block with **local losses**. It supports SCPL (`loss_fn='CL'`), DeInfo (`loss_fn='DeInfo'`), and adaptive early-exit inference.

This repository bundles the core `DecoupleFlow/` package with thesis experiment scripts under `exp/`. If you only need the library and not the experiments, use the standalone package repository: [github.com/lyt0310603/DecoupleFlow](https://github.com/lyt0310603/DecoupleFlow).

## Project Structure

```
DecoupleFlow-thesis/
├── DecoupleFlow/          # Core package (model, loss, projector, data utils)
├── exp/                   # Experiment scripts
│   ├── Amazon_review_preprocess.py
│   ├── Transformer_Amazon_BP.py
│   ├── Transformer_Amazon_SCPL.py
│   ├── NLP_exp_LSTM.py
│   ├── NLP_exp_Transformer.py
│   ├── NLP_exp_gpipe.py
│   ├── LRA_exp_haystack.py
│   ├── LRA_exp_contractnli.py
│   ├── Adaptive_exp.py
│   ├── Vision_exp.py
│   └── Vision_exp_gpipe.py
└── data/                  # Preprocessed data (e.g. amazon_review/)
```

All scripts under `exp/` add the project root to `sys.path`. Run them from the repo root:

```bash
python exp/NLP_exp_LSTM.py --help
```

## Requirements

- **Python 3.9+**
- **CUDA GPU** (recommended for training)

### Core dependencies

```bash
pip install torch torchvision numpy tqdm scikit-learn nltk pandas
pip install datasets transformers pyarrow
pip install torchgpipe   # GPipe experiments only
```

NLP scripts will download NLTK stopwords on first run.

### Hardware (by experiment type)

| Experiment | GPUs | Notes |
|------------|------|-------|
| `--arch BP` | 1 | Standard backprop on a single GPU |
| `--arch SCPL` / `DeInfo` | 4 | DecoupleFlow uses `cuda:0`–`cuda:3` |
| `*_gpipe.py` | 4 | GPipe pipeline parallelism |
| `Amazon_review_preprocess.py` | 0+ | CPU is fine; needs large disk space |

## Datasets

### NLP classification (auto-downloaded via `utils_nlp.py`)

| Dataset | Description | Scripts |
|---------|-------------|---------|
| `ag_news` | AG News (4 classes) | NLP_exp_*, Adaptive_exp |
| `imdb` | IMDB sentiment | NLP_exp_*, Adaptive_exp |
| `dbpedia_14` | DBpedia 14-class | NLP_exp_*, Adaptive_exp |

Download [IMDB](https://drive.google.com/file/d/1Z2iqiPKF5wYCgXR-Tc9ZnQqUFVkJvypA/view) and put `IMDB_Dataset.csv` in the project root.

#### GloVe embeddings

Required by LSTM-based NLP scripts (`NLP_exp_LSTM`, `NLP_exp_gpipe`, `Adaptive_exp`) via `get_word_vector()` in `utils_nlp.py`. Place `glove.6B.300d.txt` in the **project root**.

```bash
# From the repo root
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# glove.6B.300d.txt must be in the project root
```

### Long-range / LRA (`utils_nlp.py`)

| Dataset | Description | Scripts |
|---------|-------------|---------|
| `contractnli` | Contract NLI (default) | LRA_exp_haystack, LRA_exp_contractnli |
| `haystack` | Needle-in-a-Haystack (synthetic) | LRA_exp_haystack |

`haystack` is generated in code; no download is required. `LRA_exp_contractnli.py` always uses `contractnli`.

#### ContractNLI

Required by `LRA_exp_haystack.py` and `LRA_exp_contractnli.py`. Place `train.json` and `test.json` under `./contractnli/` in the **project root** (see `utils_nlp.py`).

Download from the [ContractNLI website](https://stanfordnlp.github.io/contract-nli/) (accept the terms, or use the direct link below):

```bash
# From the repo root
wget https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip
unzip contract-nli.zip
mv contract-nli contractnli
# Expected paths: ./contractnli/train.json and ./contractnli/test.json
```

| Dataset | Description | Scripts |
|---------|-------------|---------|
| `cifar10` | CIFAR-10 (torchvision auto-download) | Vision_exp, Vision_exp_gpipe |
| `cifar100` | CIFAR-100 (torchvision auto-download) | Vision_exp, Vision_exp_gpipe |
| `tinyImageNet` | Tiny ImageNet 200-class (manual download) | Vision_exp, Vision_exp_gpipe |

Download the lab-provided [tiny-imagenet-200.zip](https://drive.google.com/file/d/10wl7UjC47xuUZG5zdUwSwHP1tlV-Ubf7/view) (validation set is already pre-processed for `ImageFolder`) and extract it at the repo root as `./tiny-imagenet-200/`.

### Amazon Reviews 2023

1. Preprocess from Hugging Face:

```bash
python exp/Amazon_review_preprocess.py --output_dir ./data/amazon_review
```

2. Output: one Parquet file per category (e.g. `raw_review_Books.parquet`) for `Transformer_Amazon_SCPL.py`.

Set `AMAZON_REVIEW_OUTPUT_DIR` to override the default output path.

## Experiment CLI Reference

### Data preprocessing

#### `Amazon_review_preprocess.py`

Tokenize [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) into Parquet files.

```bash
python exp/Amazon_review_preprocess.py \
  --output_dir ./data/amazon_review \
  --tokenizer t5-base \
  --max_length 128 \
  --batch_size 10000 \
  --categories raw_review_Books
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `./data/amazon_review` | Output Parquet and logs |
| `--tokenizer` | `t5-base` | Hugging Face tokenizer |
| `--max_length` | `128` | Max sequence length |
| `--batch_size` | `10000` | Records per processing batch |
| `--categories` | all 33 | Optional subset of categories |

---

### NLP experiments

#### `NLP_exp_LSTM.py`

Two-layer BiLSTM text classification.

```bash
python exp/NLP_exp_LSTM.py \
  --dataset ag_news \
  --arch SCPL \
  --batch_size 64 \
  --epochs 50 \
  --lr 0.001 \
  --seed 42 \
  --save_path ./result/nlp_lstm.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ag_news` | `ag_news` / `imdb` / `dbpedia_14` |
| `--arch` | `BP` | `BP` / `SCPL` / `DeInfo` |
| `--batch_size` | `64` | |
| `--epochs` | `50` | |
| `--lr` | `0.001` | |
| `--max_len` | per dataset | Override default sequence length |
| `--seed` | `42` | |
| `--save_path` | `None` | JSON results path |

#### `NLP_exp_Transformer.py`

Transformer encoder for text classification.

```bash
python exp/NLP_exp_Transformer.py \
  --dataset imdb \
  --arch DeInfo \
  --batch_size 64 \
  --lr 0.0001 \
  --max_len 350 \
  --save_path ./result/nlp_transformer.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ag_news` | `ag_news` / `imdb` / `dbpedia_14` |
| `--arch` | `BP` | `BP` / `SCPL` / `DeInfo` |
| `--batch_size` | `64` | |
| `--epochs` | `50` | |
| `--lr` | `0.0001` | |
| `--max_len` | per dataset | Override default sequence length |
| `--seed` | `42` | |
| `--save_path` | `None` | JSON results path |

#### `NLP_exp_gpipe.py`

LSTM + **GPipe** baseline (4 GPUs required; no DecoupleFlow).

```bash
python exp/NLP_exp_gpipe.py \
  --dataset ag_news \
  --batch_size 64 \
  --chunks 4 \
  --epochs 50 \
  --save_path ./result/nlp_gpipe.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ag_news` | |
| `--batch_size` | `64` | |
| `--chunks` | `4` | GPipe micro-batch count |
| `--epochs` | `50` | |
| `--lr` | `0.001` | |
| `--max_len` | per dataset | |
| `--seed` | `42` | |
| `--save_path` | `None` | |

---

### Long Range Arena experiments

#### `LRA_exp_haystack.py`

LSTM on long-text tasks; reports accuracy and macro-F1.

```bash
python exp/LRA_exp_haystack.py \
  --dataset contractnli \
  --arch SCPL \
  --batch_size 32 \
  --max_len 512 \
  --save_path ./result/lra_lstm.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `BP` | `BP` / `SCPL` / `DeInfo` |
| `--batch_size` | `32` | |
| `--epochs` | `50` | |
| `--lr` | `0.001` | |
| `--max_len` | per dataset | |
| `--save_path` | `None` | |

#### `LRA_exp_contractnli.py`

Transformer on `contractnli` (fixed dataset).

```bash
python exp/LRA_exp_contractnli.py \
  --arch SCPL \
  --batch_size 512 \
  --train_classes 3 \
  --test_classes 3 \
  --save_path ./result/lra_transformer.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | `512` | |
| `--arch` | `BP` | `BP` / `SCPL` / `DeInfo` |
| `--epochs` | `50` | |
| `--lr` | `0.0001` | |
| `--max_len` | `512` | |
| `--train_classes` | `3` | 2 or 3 |
| `--test_classes` | `3` | 2 or 3 |
| `--save_path` | `None` | |

---

### Adaptive inference

#### `Adaptive_exp.py`

Deep LSTM + DecoupleFlow **adaptive early-exit** (train or load checkpoint first).

```bash
python exp/Adaptive_exp.py \
  --dataset imdb \
  --arch DeInfo \
  --batch_size 512 \
  --sim 0.8 \
  --patience 1 \
  --save_path ./result/adaptive.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `imdb` | `ag_news` / `imdb` / `dbpedia_14` |
| `--arch` | `BP` | `BP` / `SCPL` / `DeInfo` |
| `--batch_size` | `512` | |
| `--epochs` | `50` | |
| `--lr` | `0.001` | |
| `--sim` | `0.8` | Cosine similarity threshold |
| `--patience` | `1` | Early-exit patience |
| `--save_path` | `None` | |

---

### Vision experiments

#### `Vision_exp.py`

VGG / ResNet image classification.

```bash
python exp/Vision_exp.py \
  --model VGG \
  --dataset cifar10 \
  --arch SCPL \
  --batch_size 64 \
  --epochs 200 \
  --save_path ./result/vision.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `VGG` | `VGG` / `ResNet` |
| `--dataset` | `cifar10` | `cifar10` / `cifar100` / `tinyImageNet` |
| `--arch` | `BP` | `BP` / `SCPL` / `DeInfo` |
| `--batch_size` | `64` | |
| `--epochs` | `200` | |
| `--lr` | `0.001` | |
| `--seed` | `42` | |
| `--save_path` | `None` | |

#### `Vision_exp_gpipe.py`

VGG + **GPipe** baseline (4 GPUs required).

```bash
python exp/Vision_exp_gpipe.py \
  --dataset cifar10 \
  --batch_size 64 \
  --chunks 4 \
  --epochs 200 \
  --save_path ./result/vision_gpipe.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `VGG` | VGG only |
| `--dataset` | `cifar10` | |
| `--batch_size` | `64` | |
| `--chunks` | `4` | GPipe micro-batch count |
| `--epochs` | `200` | |
| `--lr` | `0.001` | |
| `--seed` | `42` | |
| `--save_path` | `None` | |

---

### Amazon Review experiments

#### `Transformer_Amazon_BP.py`

Amazon Review Transformer + single-GPU backprop baseline.

```bash
python exp/Transformer_Amazon_BP.py \
  --data_dir ./data/amazon_review \
  --category raw_review_Books \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4 \
  --save_path ./result/amazon_bp.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data/amazon_review` | Parquet directory |
| `--category` | `raw_review_Books` | Filename without `.parquet` |
| `--val_split_ratio` | `0.1` | Validation split ratio |
| `--batch_size` | `32` | |
| `--epochs` | `20` | |
| `--lr` | `1e-4` | |
| `--seed` | `42` | |
| `--num_workers` | `4` | DataLoader workers |
| `--grad_clip` | `5.0` | Gradient clipping max norm (0 to disable) |
| `--checkpoint_dir` | `None` | Optional directory for epoch checkpoints |
| `--save_path` | `None` | JSON results path |

#### `Transformer_Amazon_SCPL.py`

Amazon Review Transformer + DecoupleFlow (SCPL).

```bash
python exp/Transformer_Amazon_SCPL.py \
  --data_dir ./data/amazon_review \
  --category raw_review_Subscription_Boxes \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4 \
  --save_path ./result/amazon_scpl.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data/amazon_review` | Parquet directory |
| `--category` | `raw_review_Subscription_Boxes` | Filename without `.parquet` |
| `--val_split_ratio` | `0.1` | Validation split ratio |
| `--batch_size` | `32` | |
| `--epochs` | `20` | |
| `--lr` | `1e-4` | |
| `--seed` | `42` | |
| `--num_workers` | `4` | DataLoader workers |
| `--save_path` | `None` | |

> **Note:** `Transformer_Amazon_BP.py` is the single-GPU backprop baseline. `Transformer_Amazon_SCPL.py` uses DecoupleFlow (SCPL) across multiple GPUs.

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision numpy tqdm scikit-learn nltk pandas datasets transformers

# 2. NLP SCPL experiment
python exp/NLP_exp_LSTM.py --dataset ag_news --arch SCPL --save_path ./result/demo.json

# 3. Full Amazon pipeline
python exp/Amazon_review_preprocess.py --categories raw_review_Books
python exp/Transformer_Amazon_SCPL.py --category raw_review_Books --save_path ./result/amazon.json
```

## License

Use this code according to your thesis and project license terms.
