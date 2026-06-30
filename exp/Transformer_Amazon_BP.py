import torch
import torch.nn as nn
from Model import SCPL_model
from augment_fn import NLP_augment
from utils_fn import ResultMeter, SynchronizeTimer
import numpy as np
from tqdm import tqdm
import random
import copy
from transformers import T5Tokenizer
import random
import numpy as np
import os
import pyarrow.dataset as ds # 用於讀取多個 Parquet 檔案
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
import json
from torch.utils.data._utils.collate import default_collate
import math

# 設定隨機種子，確保實驗可重複性
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

def get_gpu(num):
    if torch.cuda.is_available():
    
        # 取得 GPU 的數量
        num_gpus = torch.cuda.device_count()
        gpus = []
        for i in range(max(num, num_gpus)):
            device_id = "cuda:" + str(i)
            gpus.append(torch.device(device_id))
    else:
        print("CUDA is not available.")
    
    return gpus

# --- 3. 建立自訂 Dataset 類別 (用於讀取 Parquet 檔案) ---
class AmazonReviewDataset(Dataset):
    def __init__(self, hf_dataset_slice):
        self.dataset = hf_dataset_slice # 這是 Hugging Face Dataset 的一個切片或子集

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx] # 獲取 Hugging Face Dataset 中的單個項目

            input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
            
            original_rating = item['label'] # 獲取原始的 'label' 值

            # --- 關鍵修改：檢查並處理無效或超出範圍的原始評分 ---
            # 判斷是否為有效標籤：必須是數字，且在 1 到 5 之間
            if (original_rating is None or 
                not isinstance(original_rating, (int, float)) or 
                not (1 <= original_rating <= 5)):
                
                # 打印警告訊息，表明這個特定樣本被忽略
                # print(f"警告: 忽略索引 {idx} 的無效標籤數據。原始值: {original_rating}")
                return None # <--- 返回 None，表示此樣本無效，DataLoader 將會自動過濾它
            
            # 如果標籤有效，進行轉換
            label = torch.tensor(int(original_rating) - 1, dtype=torch.long) # 將 1-5 轉換為 0-4

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label
            }
        
        except Exception as e:
            # 捕獲在讀取或處理單個樣本時可能發生的其他錯誤
            print(f"警告: 讀取或處理索引 {idx} 時發生異常，忽略該樣本。錯誤: {e}")
            return None # <--- 返回 None，表示此樣本無效


# 確保 collate_fn 能夠處理來自 __getitem__ 的 None 值
def custom_collate_fn(batch):
    # 步驟 1: 過濾掉批次中的所有 None 樣本
    # list(filter(...)) 會創建一個新的列表，只包含非 None 的元素
    batch = list(filter(lambda x: x is not None, batch))
    
    # 步驟 2: 如果過濾後批次為空，返回空字典
    # 這會防止 DataLoader 在所有樣本都被過濾掉時崩潰
    if len(batch) == 0:
        return {}
    
    # 步驟 3: 調用 PyTorch 預設的 default_collate 函數來處理剩餘的有效樣本
    # default_collate 會自動處理張量、數字、列表、字典等的堆疊
    return default_collate(batch)

def get_dataloader(DATA_DIR, VAL_SPLIT_RATIO, SEED, BATCH_SIZE):
    # --- 1. 定義要載入的特定商品類別檔案 (單一檔案) ---
    # selected_category_file = "raw_review_Subscription_Boxes.parquet"
    # selected_category_file = "raw_review_Software.parquet"
    selected_category_file = "raw_review_Books.parquet"
    # selected_category_file = "raw_review_Clothing_Shoes_and_Jewelry.parquet"
    selected_parquet_path = os.path.join(DATA_DIR, selected_category_file)

    print(f"載入選定的 Amazon Reviews 數據 (單一檔案: {selected_category_file} 從 {DATA_DIR})...")
    try:
        # 只從選定的單一檔案路徑載入數據
        full_hf_dataset = HFDataset.from_parquet(
            selected_parquet_path, # <--- 直接傳遞單一檔案路徑
            cache_dir=None, # 保留這個，你之前處理過這個參數的問題
        )
        print(f"成功載入選定商品類別數據，總計 {len(full_hf_dataset)} 條記錄。")
    except Exception as e:
        print(f"從 Parquet 載入數據失敗: {e}")
        print("請確認選定 Parquet 檔案路徑正確且格式完整。")
        exit() # 載入失敗則退出腳本
    
    # --- 2. 劃分訓練集和驗證集 (這部分邏輯不變) ---
    # 因為現在數據量較小，就不再進行額外的分層取樣，直接劃分訓練和驗證
    shuffled_dataset = full_hf_dataset.shuffle(seed=SEED)
    total_size = len(shuffled_dataset)
    val_size = int(total_size * VAL_SPLIT_RATIO)
    train_size = total_size - val_size
    
    train_hf_dataset = shuffled_dataset.select(range(train_size))
    eval_hf_dataset = shuffled_dataset.select(range(train_size, total_size))
    
    # 建立自訂 Dataset 實例
    train_dataset = AmazonReviewDataset(train_hf_dataset)
    eval_dataset = AmazonReviewDataset(eval_hf_dataset)
    
    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(eval_dataset)}")

    # --- 3. 建立 DataLoader ---
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        generator=g,  # 使用固定的隨機種子
        pin_memory=True, # 如果有 GPU 建議設為 True
        num_workers=4, # 可以根據 CPU 核心數調整，如果遇到 DataLoader worker 錯誤，先設為 0 測試
        collate_fn=custom_collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE,
        generator=g,  # 驗證集也使用相同的隨機種子
        pin_memory=True,
        num_workers=4, # 同上
        collate_fn=custom_collate_fn
    )
    
    print(f"訓練 DataLoader steps: {len(train_loader)}")
    print(f"驗證 DataLoader steps: {len(eval_loader)}")
    return train_loader, eval_loader, 5

def accuracy(output, target):
     with torch.no_grad():
        bsz = target.shape[0]
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = correct[0].view(-1).float().sum(0, keepdim=True).mul_(100 / bsz)
        return acc


def train_model(model, train_loader, epoch, num_epochs, optimizer, criterion, device):
    losses = ResultMeter()
    train_time = ResultMeter()
    eval_time = ResultMeter()
    eval_acc = ResultMeter()
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Training")
    for step, batch in enumerate(loop):
        if not batch or batch['input_ids'].shape[0] == 0:
            continue
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        model.train()
        
        with SynchronizeTimer() as train_timer:
            logits = model(input_ids, attention_mask) # 模型現在直接返回 logits
            loss = criterion(logits, labels) # 計算損失  
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # max_norm 是超參數，通常設為 0.5 或 1.0 或 5.0
            optimizer.step()

        losses.update(loss.item())
        train_time.update(train_timer.runtime)
        loop.set_postfix(loss=loss.item())

        correct = 0
        total = 0
        model.eval()
        with SynchronizeTimer() as eval_timer:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                acc = accuracy(outputs, labels)
        eval_acc.update(acc.item())
        eval_time.update(eval_timer.runtime)  
            
    # return losses.avg, train_time.sum, eval_acc.avg, eval_time.sum, lr
    return losses.avg, train_time.sum, eval_acc.avg, eval_time.sum

def test_model(model, testloader):
    test_time = ResultMeter()
    model.eval()
    accs = ResultMeter()
    eval_acc = ResultMeter()

    for batch in testloader:
        if not batch or batch['input_ids'].shape[0] == 0:
            continue
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with SynchronizeTimer() as test_timer:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                acc = accuracy(outputs, labels)
                accs.update(acc.item())
        test_time.update(test_timer.runtime)

    # return test_time.sum, accs.avg
    return test_time.sum, accs.avg


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 創建一個位置嵌入層，它就是一個可學習的查找表
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x 的維度: (batch_size, seq_len, d_model) """
        # 1. 獲取輸入的序列長度
        seq_len = x.size(1)

        # 2. 創建位置 ID (從 0 到 seq_len - 1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        # 3. 從位置嵌入層中查找對應的位置向量
        position_embeddings = self.pos_embedding(position_ids)
        
        # 4. 將位置嵌入與詞嵌入相加 (利用廣播機制)
        x = x + position_embeddings
        return self.dropout(x)

# 定義 Transformer 分類器類別
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, nhead, dim_feedforward):
        super().__init__()

        # Embedding Layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Transformer Layers
        # self.encoder1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.encoder1 = nn.TransformerEncoderLayer(
                            d_model=hidden_size,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            batch_first=True
                        )
        self.encoder2 = nn.TransformerEncoderLayer(
                            d_model=hidden_size,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            batch_first=True
                        )
        self.encoder3 = nn.TransformerEncoderLayer(
                            d_model=hidden_size,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            batch_first=True
                        )
        
        # 線性層和激活函數
        self.linear1 = nn.Linear(hidden_size, 300)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(300, num_classes) # 分類層
        self.d_model = embedding_dim

    def forward(self, input_ids, attention_mask):
        # 嵌入層
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)

        # 準備 padding mask
        padding_mask = (attention_mask == 0)
        
        # Encoder 層
        embedding_output = embedded
        transformer_output1 = self.encoder1(src=embedding_output, src_key_padding_mask=padding_mask)
        transformer_output2 = self.encoder2(src=transformer_output1, src_key_padding_mask=padding_mask)
        transformer_output3 = self.encoder3(src=transformer_output2, src_key_padding_mask=padding_mask)
        
        # 池化輸出
        pooled_output = transformer_output3.mean(dim=1)
        
        # 線性層和激活
        linear_out = self.linear1(pooled_output)
        tanh_out = self.tanh(linear_out)
        
        # 分類層
        logits = self.classifier(tanh_out) # shape: (batch_size, num_classes)
        
        return logits


if __name__ == '__main__':
    output_dir = "./model_checkpoints" # 儲存模型的目錄
    os.makedirs(output_dir, exist_ok=True) # 確保目錄存在
    # 設定隨機種子
    SEED = 42
    set_seed(SEED)

    epochs = 20
    
    DATA_DIR = "/work/lyt0310603/amazon_review"
    VAL_SPLIT_RATIO = 0.1
    BATCH_SIZE = 32
    train_loader, test_loader, n_classes = get_dataloader(DATA_DIR, VAL_SPLIT_RATIO, SEED, BATCH_SIZE)
    
    embed_size = 512
    hidden_size = 512
    nhead = 8
    dim_feedforward = 2048
    # embed_size = 128
    # hidden_size = 128
    # nhead = 4
    # dim_feedforward = 512
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    vocab_size = tokenizer.vocab_size
    
    model = TransformerEncoderClassifier(
        vocab_size=vocab_size,
        embedding_dim=embed_size,
        hidden_size=hidden_size,
        num_classes=n_classes,
        nhead=nhead,
        dim_feedforward=dim_feedforward
    )
    
    # 選擇設備並將模型移動到設備
    device = get_gpu(1) # 獲取第一個 GPU 設備
    device = device[0]
    model.to(device)

    # 定義優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) 
    
    # 定義損失函數
    criterion = torch.nn.CrossEntropyLoss()
    
    print(model)
    loss = []
    train_time = []
    eval_time = []
    train_acc = []
    train_classifier_accs = []
    test_classifier_accs = []
    
    test_time = []
    test_acc = []
    report = {}
    for i in range(1, epochs+1):
        r_loss, r_train_time, r_train_acc, r_eval_time = train_model(model, train_loader, i, epochs, optimizer, criterion, device)
    
        print(f'epoch: {i}')
        print(f'T_Time: {r_train_time:.2f}, E_Time: {r_eval_time:.2f}, loss: {r_loss:.3f}, Acc: {r_train_acc:.2f}')
        loss.append(r_loss)
        train_time.append(r_train_time)
        train_acc.append(r_train_acc)
        eval_time.append(r_eval_time)
        
        r_test_time, r_test_acc = test_model(model, test_loader)
        print(f'T_Time: {r_test_time:.2f}, Acc: {r_test_acc:.2f}')
        test_time.append(r_test_time)
        test_acc.append(r_test_acc)
        
        checkpoint_path = os.path.join(output_dir, f"Transformer_BP_model_epoch_{i}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"模型權重已儲存到: {checkpoint_path}")
        
        for i in range(0, len(train_time)):
            report[str(i+1)] = {
                "epoch": i+1,
                "train acc": train_acc[i],
                "train loss": loss[i],
                "train time": train_time[i],
                "train eval. time": eval_time[i],
                "test acc": test_acc[i],
                "test time": test_time[i],
                
            }
        with open("Transformer_amazon_BP.json", "w") as json_file:
            json.dump(report, json_file, indent=4)
