"""Modified from https://github.com/Hibb-bb/AL"""

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset as HFDataset
import json
import os
import re
import string
import itertools
from collections import Counter
from tqdm import tqdm
import numpy as np
import io
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from datasets import load_dataset, ClassLabel, Value
from torch.utils.data import DataLoader
import random

# ContractNLI 帶狀過濾參數（以「最早 gold evidence 的絕對 token 位置」為準，
# token 位置用與訓練一致的前處理 remove_stopword=False 計算）：
#   - 下限 CONTRACTNLI_MIN_EVIDENCE_TOKENS：證據位置 < 下限的 E/C 樣本會被移除
#     （這些「證據在前段」的樣本短 max_len 就能答對，會壓平長度趨勢）。
#   - 上限 CONTRACTNLI_MAX_EVIDENCE_TOKENS：證據位置 > 上限的樣本會被移除
#     （這些證據超出實驗最大長度、永遠抓不到，是死重）。0 代表不設上限。
# 只影響 entailment/contradiction；notmentioned 不受此過濾。可由 get_data 的
# args['evidence_min'] / args['evidence_max'] 覆蓋（預設沿用下方常數）。
CONTRACTNLI_MIN_EVIDENCE_TOKENS = 0
CONTRACTNLI_MAX_EVIDENCE_TOKENS = 0

def get_data(args):
    
    if args['dataset'] == 'haystack':
        print("[INFO] Loading 'hay_stack' (Needle in Haystack) dataset.")
        # 您的實驗列表
        experimental_lengths_list = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050]
        
        # 分佈參數
        mu = 675 
        sigma = 200
        class_num = 10
        vocab_size = 15 # (這個值現在不太重要了，但 create_vocab 需要它)

        # 2. 生成資料 (移除 real_haystack_words 參數)
        train_text, train_label = generate_needle_haystack_samples(
            num_samples=24000,
            total_len=max(experimental_lengths_list),
            class_num=class_num,
            needle_pos_mu=mu,
            needle_pos_sigma=sigma
        )
        test_text, test_label = generate_needle_haystack_samples(
            num_samples=2400,
            total_len=max(experimental_lengths_list),
            class_num=class_num,
            needle_pos_mu=mu,
            needle_pos_sigma=sigma
        )

        # 3. 資料是人造的，不需 data_preprocessing 或 data_cleansing
        clean_train = train_text
        clean_test = test_text
        
        # 4. 建立詞彙表
        # vocab 現在只會包含 'magic', 'needle', 'key', 'valX' 和 '<pad>'
        vocab = create_vocab(clean_train, vocab_size=vocab_size)
        
        # 5. 手動映射特殊 token
        key_tokens = ["magic", "needle", "key"]
        val_tokens = [f"val{i}" for i in range(class_num)]
        
        # === 修正ID衝突錯誤 ===
        special_id_counter = len(vocab) # <--- 從 len(vocab) 開始
        
        for token in key_tokens + val_tokens:
            if token not in vocab: 
                vocab[token] = special_id_counter
                special_id_counter += 1
        print(f"[INFO] Special 'hay_stack' tokens mapped in vocab.")
        
        # 6. 建立 Textset 和 DataLoader
        trainset = Textset(clean_train, train_label, vocab, args['max_len'])
        testset = Textset(clean_test, test_label, vocab, args['max_len'])
        
        train_loader = DataLoader(
            trainset, batch_size=args['train_bsz'], collate_fn=trainset.collate, shuffle=True, pin_memory=True)
        test_loader = DataLoader(
            testset, batch_size=args['test_bsz'], collate_fn=testset.collate, pin_memory=True)
        
        if float(args['noise_rate']) != 0:
            add_noise(train_loader, class_num, float(args['noise_rate']))
    
        # 7. === 關鍵：在此 return，繞過後續的通用 pre-processing ===
        return train_loader, test_loader, class_num, vocab
        
    if args['dataset'] != 'imdb':
        
        if args['dataset'] == 'emotion_semisynth':
            print("[INFO] Loading 'emotion_semisynth' dataset.")
            train_path = './emotion_semisynth/train.jsonl'
            test_path = './emotion_semisynth/test.jsonl'
            class_num = 6

            if not os.path.exists(train_path):
                raise FileNotFoundError("Cannot find emotion_semisynth train file: {}".format(train_path))
            if not os.path.exists(test_path):
                raise FileNotFoundError("Cannot find emotion_semisynth test file: {}".format(test_path))

            def _load_jsonl(path):
                texts = []
                labels = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        row = json.loads(line)
                        text = row.get('text', '')
                        label = row.get('label', None)
                        if len(text) == 0 or label is None:
                            continue
                        texts.append(text)
                        labels.append(int(label))
                return texts, labels

            train_text, train_label = _load_jsonl(train_path)
            test_text, test_label = _load_jsonl(test_path)
            clean_train = [data_preprocessing(t, True) for t in train_text]
            clean_test = [data_preprocessing(t, True) for t in test_text]
            clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
            clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)
            vocab = create_vocab(clean_train)

        elif args['dataset'] == 'contractnli':
            print("[INFO] Loading 'ContractNLI' dataset for 3-way classification.")
            train_path = './contractnli/train.json'
            test_path = './contractnli/test.json'
            class_num = 3

            if not os.path.exists(train_path):
                raise FileNotFoundError("Cannot find ContractNLI train file: {}".format(train_path))
            if not os.path.exists(test_path):
                raise FileNotFoundError("Cannot find ContractNLI test file: {}".format(test_path))

            label_map = {
                'entailment': 0,
                'contradiction': 1,
                'notmentioned': 2
            }

            # 帶狀過濾門檻（token 位置）：優先用 args 覆蓋，否則用模組常數。
            evidence_min = int(args.get('evidence_min', CONTRACTNLI_MIN_EVIDENCE_TOKENS) or 0)
            evidence_max_raw = args.get('evidence_max', CONTRACTNLI_MAX_EVIDENCE_TOKENS)
            evidence_max = int(evidence_max_raw) if evidence_max_raw and int(evidence_max_raw) > 0 else None

            def _earliest_evidence_token_pos(hypothesis, doc_text, prefix_text):
                # 與訓練一致：combined = "hypothesis sep doc"，前處理保留停用詞。
                hyp_tokens = len(data_preprocessing(hypothesis, False).split())
                prefix_tokens = len(data_preprocessing(prefix_text, False).split())
                return hyp_tokens + 1 + prefix_tokens  # +1 為分隔詞 "sep"

            def _load_contract_nli(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                all_labels = data.get("labels", {})
                documents = data.get("documents", [])
                texts = []
                labels = []
                filtered_too_early = 0
                filtered_too_late = 0
                filtered_missing_evidence = 0

                for doc in documents:
                    doc_text = doc.get("text", "")
                    doc_len = len(doc_text)
                    annotation_sets = doc.get("annotation_sets", [])
                    if not annotation_sets:
                        continue
                    annotations = annotation_sets[0].get("annotations", {})

                    for hyp_key, annot in annotations.items():
                        raw_choice = str(annot.get("choice", "")).strip().lower()
                        if raw_choice not in label_map:
                            continue

                        hypothesis = all_labels.get(hyp_key, {}).get("hypothesis", "")
                        if len(hypothesis) == 0 or len(doc_text) == 0:
                            continue

                        # NotMentioned 不需要 evidence；另外兩類用 gold evidence 的
                        # 絕對 token 位置做帶狀過濾。
                        if raw_choice != 'notmentioned':
                            spans = annot.get("spans", [])
                            valid_span_indices = []
                            if isinstance(spans, list):
                                valid_span_indices = [
                                    span_idx for span_idx in spans
                                    if isinstance(span_idx, int) and 0 <= span_idx < len(doc.get("spans", []))
                                ]
                            if len(valid_span_indices) == 0:
                                filtered_missing_evidence += 1
                                continue

                            doc_spans = doc.get("spans", [])
                            earliest_char_start = None
                            for span_idx in valid_span_indices:
                                span = doc_spans[span_idx]
                                if isinstance(span, list) and len(span) >= 2 and isinstance(span[0], int):
                                    char_start = max(0, min(span[0], doc_len))
                                    if earliest_char_start is None or char_start < earliest_char_start:
                                        earliest_char_start = char_start
                            if earliest_char_start is None:
                                continue

                            earliest_token_pos = _earliest_evidence_token_pos(
                                hypothesis, doc_text, doc_text[:earliest_char_start]
                            )
                            if earliest_token_pos < evidence_min:
                                filtered_too_early += 1
                                continue
                            if evidence_max is not None and earliest_token_pos > evidence_max:
                                filtered_too_late += 1
                                continue

                        combined_text = "{} sep {}".format(hypothesis, doc_text)
                        texts.append(combined_text)
                        labels.append(label_map[raw_choice])

                print("[INFO] ContractNLI {} | kept={} | filtered_too_early={} | filtered_too_late={} | filtered_missing_evidence={} | evidence_min={} | evidence_max={}".format(
                    file_path, len(texts), filtered_too_early, filtered_too_late,
                    filtered_missing_evidence, evidence_min, evidence_max
                ))
                return texts, labels

            train_text, train_label = _load_contract_nli(train_path)
            test_text, test_label = _load_contract_nli(test_path)
            # ContractNLI 是 NLI 任務，否定詞 (not/no/nor...) 對判斷蘊含 vs 矛盾
            # 至關重要，因此「不」移除停用詞，避免砍掉決定性線索。
            clean_train = [data_preprocessing(t, False) for t in train_text]
            clean_test = [data_preprocessing(t, False) for t in test_text]
            clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
            clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)
            vocab = create_vocab(clean_train)

        else:
            if args['dataset'] == 'hyperpartisan':
                full_dataset = load_dataset('hyperpartisan_news_detection', "bypublisher", split='train[:10%]')
                full_dataset = full_dataset.rename_column("hyperpartisan", "label")
                full_dataset = full_dataset.cast_column("label", ClassLabel(num_classes=2, names=['false', 'true']))
                dataset_split = full_dataset.train_test_split(test_size=0.2)
                train_data = dataset_split['train']
                test_data = dataset_split['test']
            else:
                train_data = load_dataset(args['dataset'], split='train')
                test_data = load_dataset(args['dataset'], split='test')

            if args['dataset'] == 'dbpedia_14':
                tf = 'content'
                class_num = 14
            elif args['dataset'] == 'ag_news':
                tf = 'text'
                class_num = 4
            elif args['dataset'] == 'banking77':
                tf = 'text'
                class_num = 77
            elif args['dataset'] == 'emotion':
                tf = 'text'
                class_num = 6
            elif args['dataset'] == 'rotten_tomatoes':
                tf = 'text'
                class_num = 2
            elif args['dataset'] == 'yelp_review_full':
                tf = 'text'
                class_num = 5
            elif args['dataset'] == 'sst2':
                tf = 'sentence'
                class_num = 2
                test_data = load_dataset(args['dataset'], split='validation')
            elif args['dataset'] == 'hyperpartisan':
                tf = 'text'
                class_num = 2
                class_num = 20
            else:
                raise ValueError("Dataset not supported: {}".format(args['dataset']))

            train_text = [b[tf] for b in train_data]
            test_text = [b[tf] for b in test_data]
            train_label = [b['label'] for b in train_data]
            test_label = [b['label'] for b in test_data]
            clean_train = [data_preprocessing(t, True) for t in train_text]
            clean_test = [data_preprocessing(t, True) for t in test_text]
            clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
            clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)

            vocab = create_vocab(clean_train)

    else:
        from sklearn.model_selection import train_test_split
        class_num = 2
        df = pd.read_csv('./IMDB_Dataset.csv')
        df['cleaned_reviews'] = df['review'].apply(data_preprocessing, True)
        # df['cleaned_reviews'] = df['review'].apply(data_preprocessing, False)
        corpus = [word for text in df['cleaned_reviews']
                  for word in text.split()]
        text = [t for t in df['cleaned_reviews']]
        label = []
        for t in df['sentiment']:
            if t == 'negative':
                label.append(1)
            else:
                label.append(0)
        vocab = create_vocab(corpus)
        clean_train, clean_test, train_label, test_label = train_test_split(
            text, label, test_size=0.2)
        clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
        clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)
        
    trainset = Textset(clean_train, train_label, vocab, args['max_len'])
    testset = Textset(clean_test, test_label, vocab, args['max_len'])
    
    train_loader = DataLoader(
        trainset, batch_size=args['train_bsz'], collate_fn=trainset.collate, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        testset, batch_size=args['test_bsz'], collate_fn=testset.collate, pin_memory=True)
    
    if float(args['noise_rate']) != 0:
        add_noise(train_loader, class_num, float(args['noise_rate']))

    return train_loader, test_loader, class_num, vocab

def get_word_vector(vocab, emb='glove'):

    if emb == 'glove':
        fname = 'glove.6B.300d.txt'

        with open(fname, 'rt', encoding='utf8') as fi:
            full_content = fi.read().strip().split('\n')

        data = {}
        for i in tqdm(range(len(full_content)), total=len(full_content), desc='loading glove vocabs...'):
            i_word = full_content[i].split(' ')[0]
            if i_word not in vocab.keys():
                continue
            i_embeddings = [float(val)
                            for val in full_content[i].split(' ')[1:]]
            data[i_word] = i_embeddings

    elif emb == 'fasttext':
        fname = 'wiki-news-300d-1M.vec'

        fin = io.open(fname, 'r', encoding='utf-8',
                      newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}

        for line in tqdm(fin, total=1000000, desc='loading fasttext vocabs...'):
            tokens = line.rstrip().split(' ')
            if tokens[0] not in vocab.keys():
                continue
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)

    else:
        raise Exception('emb not implemented')

    w = []
    find = 0
    for word in vocab.keys():
        try:
            w.append(torch.tensor(data[word]))
            find += 1
        except:
            w.append(torch.rand(300))

    print('found', find, 'words in', emb)
    return torch.stack(w, dim=0)

def data_cleansing(_text, _labels, doRemove=False):
    """
    Detect or remove the empty samples.
    """
    assert len(_text)==len(_labels), "Text and label list need to be the same length."

    clear_text = []
    clear_label = []
    flag = False

    for idx ,t in enumerate(_text):
        if len(t) == 0:
            flag = True
        else:
            if doRemove:
                clear_text.append(t)
                clear_label.append(_labels[idx])

    if (flag == True) and (doRemove == True):
        print("Info: Detect the empty samples, and remove them!")
        print("Size change: {0}->{1}".format(len(_text), len(clear_text)))
    elif (flag == True) and (doRemove == False):
        print("Warning: same samples in data preprocessing outputs empty list. This will damage the model.")

    if doRemove:
        return clear_text, clear_label
    else:
        return _text, _labels


def data_preprocessing(text, remove_stopword=False):

    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]
    text = ' '.join(text)

    return text

def create_vocab(corpus, vocab_size=30000):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    # 1. 立即初始化 <pad> 和 <unk>，確保它們的 ID 是 0 和 1
    vocab_to_int = {'<pad>': 0, '<unk>': 1}
    current_id = 2 # 新單字從 ID 2 開始

    # 2. 遍歷排序後的單字
    for w, c in sorted_words:
        # 3. 如果單字不是我們已經手動添加的
        if w not in vocab_to_int:
            vocab_to_int[w] = current_id
            current_id += 1
        
        # 4. 檢查是否達到了總大小限制
        if len(vocab_to_int) >= vocab_size:
            break
    # --- 修正結束 ---
            
    print('vocab size', len(vocab_to_int))
    return vocab_to_int

def add_noise(loader, class_num, noise_rate):
    """ Referenced from https://github.com/PaulAlbert31/LabelNoiseCorrection """
    print("[DATA INFO] Use noise rate {} in training dataset.".format(float(noise_rate)))
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.y]
    text = [sample_i for sample_i in loader.sampler.data_source.x]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_rate*100)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(set(range(class_num)))
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    # loader.sampler.data_source.x = text
    loader.sampler.data_source.y = noisy_labels

    return noisy_labels

class Textset(Dataset):
    def __init__(self, text, label, vocab, max_len, pad_value=0, pad_token='<pad>'):
        super().__init__()
        self.pad_value = pad_value
        self.pad_token = pad_token

        method = 1
        self.handle(text, label, vocab, max_len, method)

    def handle(self, text, label, vocab, max_len, method=1):

        if method == 0:
            print("[Textset] Using method 0")
            new_text = []
            for t in text:
                t_split = t.split(' ')
                if len(t_split) > max_len:
                    t_split = t_split[:max_len]
                    new_text.append(' '.join(t_split))
                else:
                    while len(t_split) < max_len:
                        t_split.append(self.pad_token)
                    new_text.append(' '.join(t_split))
            self.x = new_text
            self.y = label
            self.vocab = vocab
        
        elif method == 1:
            print("[Textset] Using method 1")
            new_text = []
            for t in text:
                t_split = t.split(' ')
                if len(t_split) > max_len:
                    t_split = t_split[:max_len]
                    new_text.append(' '.join(t_split))
                else:
                    new_text.append(' '.join(t_split))
            self.x = new_text
            self.y = label
            self.vocab = vocab

        elif method == 2:
            print("[Textset] Using method 2")
            new_text = []
            for t in text:
                if len(t) > max_len:
                    t = t[:max_len]
                    new_text.append(t)
                else:
                    new_text.append(t)
            self.x = new_text
            self.y = label
            self.vocab = vocab
        else:
            raise RuntimeError("Textset method setting error!")

    def collate(self, batch):
        x = [torch.tensor(x) for x, y in batch]
        y = [y for x, y in batch]
        x_tensor = pad_sequence(x, True)
        mask = (x_tensor == 0)
        y = torch.tensor(y)
        return x_tensor, y, mask

    def convert2id(self, text):
        r = []
        for word in text.split():
            if word in self.vocab.keys():
                r.append(self.vocab[word])
            else:
                r.append(self.vocab['<unk>'])
        return r

    def __getitem__(self, idx):
        text = self.x[idx]
        word_id = self.convert2id(text)
        return word_id, self.y[idx]

    def __len__(self):
        return len(self.x)

def generate_needle_haystack_samples(num_samples, total_len, class_num, 
                                     needle_pos_mu, needle_pos_sigma):
    """
    生成 "大海撈針" 任務的樣本。
    "針" 的位置會從一個常態分佈 (mu, sigma) 中抽樣。
    "稻草" 將使用 <pad> (ID 0) 填充，以強制模型學習 "針" 的信號。
    """
    
    # --- 關鍵參數設定 ---
    # 1. 針的關鍵字
    key_tokens = ["magic", "needle", "key"] 
    val_tokens = [f"val{i}" for i in range(class_num)]
    needle_len = len(key_tokens) + 1 # (e.g., "magic needle key val7")
    
    # 2. "稻草" (*** 關鍵修改 ***)
    # 我們不再使用 GloVe 單字，而是使用 <pad> token
    # 您的 vocab['<pad>'] = 0
    # 您的 Textset.convert2id 會將 <pad> 轉為 ID 0
    HAYSTACK_TOKEN = "<pad>" 
    
    print(f"[INFO] Generating {num_samples} 'hay_stack' samples | TotalLen={total_len}")
    print(f"[INFO] Haystack will be filled with '{HAYSTACK_TOKEN}' token.")
    print(f"[INFO] Needle positions sampled from: Normal(mu={needle_pos_mu}, sigma={needle_pos_sigma})")
    
    all_texts = []
    all_labels = []
    
    # 3. 一次性生成所有 "針" 的位置
    sampled_positions = np.random.normal(loc=needle_pos_mu, 
                                         scale=needle_pos_sigma, 
                                         size=num_samples)
    
    # 4. 確保位置在合理範圍內 (截斷)
    min_pos = 10 
    max_pos = total_len - needle_len 
    sampled_positions = np.clip(sampled_positions, min_pos, max_pos).astype(int)
    
    for i in range(num_samples):
        # 5. 生成 "稻草" (*** 關鍵修改 ***)
        # 建立一個全 <pad> 的 list
        haystack = [HAYSTACK_TOKEN] * total_len
        
        # 6. 生成標籤 (答案)
        label = random.randint(0, class_num - 1)
        
        # 7. 建立 "針"
        needle = key_tokens + [val_tokens[label]]
        
        # 8. 獲取這筆樣本的 "針" 位置
        chosen_pos = sampled_positions[i]
        
        # 9. 插入 "針"
        haystack[chosen_pos : chosen_pos + len(needle)] = needle
        
        # 10. 轉回字串
        all_texts.append(" ".join(haystack))
        all_labels.append(label)
        
    return all_texts, all_labels