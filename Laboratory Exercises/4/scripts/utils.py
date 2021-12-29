import os

import pandas as pd
from tqdm import tqdm

def tokenize(X, tokenizer=None):
    if not tokenizer:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    input_ids, attention_masks = [], []
    for sentence in tqdm(X):
        sentence_tokens = tokenizer.encode_plus(sentence, max_length=10, padding='max_length', truncation=True)
        input_ids.append(sentence_tokens["input_ids"])
        attention_masks.append(sentence_tokens['attention_mask'])

    return input_ids, attention_masks

def load_data() -> pd.DataFrame:
    """
    Loads the data from the csv file.
    :return: DataFrame
    """
    data_dir = os.getenv("DATA_DIR") or "data"
    data_file = os.getenv("DATA_FILE") or "trial.csv"
    return pd.read_csv(os.path.join(data_dir, data_file), delimiter="	")