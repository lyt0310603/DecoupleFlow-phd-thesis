import argparse
import gc
import logging
import os
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import T5Tokenizer

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    'data',
    'amazon_review',
)

ALL_REVIEW_CONFIGS = [
    'raw_review_All_Beauty', 'raw_review_Toys_and_Games', 'raw_review_Cell_Phones_and_Accessories',
    'raw_review_Industrial_and_Scientific', 'raw_review_Gift_Cards', 'raw_review_Musical_Instruments',
    'raw_review_Electronics', 'raw_review_Handmade_Products', 'raw_review_Arts_Crafts_and_Sewing',
    'raw_review_Baby_Products', 'raw_review_Health_and_Household', 'raw_review_Office_Products',
    'raw_review_Digital_Music', 'raw_review_Grocery_and_Gourmet_Food', 'raw_review_Sports_and_Outdoors',
    'raw_review_Home_and_Kitchen', 'raw_review_Subscription_Boxes', 'raw_review_Tools_and_Home_Improvement',
    'raw_review_Pet_Supplies', 'raw_review_Video_Games', 'raw_review_Kindle_Store',
    'raw_review_Clothing_Shoes_and_Jewelry', 'raw_review_Patio_Lawn_and_Garden', 'raw_review_Unknown',
    'raw_review_Books', 'raw_review_Automotive', 'raw_review_CDs_and_Vinyl',
    'raw_review_Beauty_and_Personal_Care', 'raw_review_Amazon_Fashion', 'raw_review_Magazine_Subscriptions',
    'raw_review_Software', 'raw_review_Health_and_Personal_Care', 'raw_review_Appliances',
    'raw_review_Movies_and_TV',
]


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess Amazon Reviews 2023 into Parquet files.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.environ.get('AMAZON_REVIEW_OUTPUT_DIR', DEFAULT_OUTPUT_DIR),
        help='Directory for output Parquet files and logs.',
    )
    parser.add_argument('--tokenizer', type=str, default='t5-base', help='Hugging Face tokenizer name.')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum token sequence length.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Processing batch size per category.')
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        choices=ALL_REVIEW_CONFIGS,
        metavar='CATEGORY',
        help='Optional subset of review categories to process. Defaults to all categories.',
    )
    return parser.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file_name = f"amazon_review_preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("--- 腳本開始執行 ---")
    logger.info(f"日誌檔案路徑: {log_path}")
    return logger


def build_schema(max_length):
    return pa.schema([
        pa.field('input_ids', pa.list_(pa.int32(), max_length)),
        pa.field('attention_mask', pa.list_(pa.int32(), max_length)),
        pa.field('label', pa.int8()),
    ])


def process_category(config_name, tokenizer, output_dir, max_length, batch_size, schema, logger):
    parquet_file_path = os.path.join(output_dir, f"{config_name}.parquet")

    if os.path.exists(parquet_file_path):
        logger.info(f"檔案 {parquet_file_path} 已存在，跳過此類別。")
        return

    writer = None
    current_dataset = None
    try:
        logger.info(f"載入數據集: {config_name}...")
        current_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            config_name,
            split="full",
            trust_remote_code=True,
        )
        num_records_in_category = len(current_dataset)
        logger.info(f"數據集 {config_name} 包含 {num_records_in_category} 條評論。")

        writer = pq.ParquetWriter(parquet_file_path, schema, compression='ZSTD')

        for i in range(0, num_records_in_category, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, num_records_in_category)

            batch_data_list = current_dataset.select(range(batch_start, batch_end)).to_list()

            texts = [item['text'] if item['text'] is not None else "" for item in batch_data_list]
            labels = [int(item['rating']) if item['rating'] is not None else 0 for item in batch_data_list]

            tokenized_inputs = tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="np",
            )

            input_ids_arrow = pa.array(
                tokenized_inputs['input_ids'].tolist(),
                type=pa.list_(pa.int32(), max_length),
            )
            attention_mask_arrow = pa.array(
                tokenized_inputs['attention_mask'].tolist(),
                type=pa.list_(pa.int32(), max_length),
            )
            labels_arrow = pa.array(labels, type=pa.int8())

            table = pa.Table.from_arrays(
                [input_ids_arrow, attention_mask_arrow, labels_arrow],
                names=['input_ids', 'attention_mask', 'label'],
            )

            writer.write_table(table)
            logger.info(
                f"  已處理並寫入 {config_name} 的批次: {batch_start}-{batch_end}/{num_records_in_category}"
            )

            del batch_data_list, texts, labels, tokenized_inputs, input_ids_arrow, attention_mask_arrow, labels_arrow, table
            gc.collect()

        logger.info(f"--- 商品類別 {config_name} 處理完畢，儲存到 {parquet_file_path} ---")

    except Exception as e:
        logger.error(f"處理商品類別 {config_name} 時發生錯誤: {e}", exc_info=True)
        logger.error(f"錯誤類型: {type(e).__name__}, 錯誤訊息: {str(e)}")
        logger.error("請檢查錯誤訊息並確保有足夠的記憶體和磁碟空間。")
    finally:
        if writer is not None:
            writer.close()
        if current_dataset is not None:
            del current_dataset
        gc.collect()


def main():
    args = get_args()
    output_dir = os.path.abspath(args.output_dir)
    review_configs = args.categories or ALL_REVIEW_CONFIGS

    logger = setup_logging(output_dir)
    logger.info(f"TOKENIZER: {args.tokenizer}")
    logger.info(f"MAX_LENGTH: {args.max_length}")
    logger.info(f"OUTPUT_DIR: {output_dir}")
    logger.info(f"BATCH_SIZE: {args.batch_size}")
    logger.info(f"CATEGORIES: {len(review_configs)}")

    logger.info(f"載入 Tokenizer: {args.tokenizer}...")
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer pad_token 設定為: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    schema = build_schema(args.max_length)
    logger.info("PyArrow Schema 已定義。")

    for config_name in review_configs:
        logger.info(f"\n--- 開始處理商品類別: {config_name} ---")
        process_category(
            config_name,
            tokenizer,
            output_dir,
            args.max_length,
            args.batch_size,
            schema,
            logger,
        )

    logger.info("\n所有商品類別的評論數據處理完畢。")
    logger.info(f"所有處理後的 Parquet 檔案儲存在: {output_dir} 目錄下。")
    logger.info("--- 腳本執行結束 ---")


if __name__ == '__main__':
    main()
