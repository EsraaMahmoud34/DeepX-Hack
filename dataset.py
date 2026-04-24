import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import ASPECTS, SENTIMENT_LABELS
from preprocessing import preprocess_text

def encode_aspects(aspect_list):
    return [1 if aspect in aspect_list else 0 for aspect in ASPECTS]

def load_data(path, is_inference=False):
    df = pd.read_excel(path)

    if not is_inference:
        # Convert columns if they exist
        if "aspects" in df.columns:
            df["aspects"] = df["aspects"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        if "aspect_sentiments" in df.columns:
            df["aspect_sentiments"] = df["aspect_sentiments"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
        df["aspect_labels"] = df["aspects"].apply(encode_aspects)

    # Apply text preprocessing
    df["review_text"] = df["review_text"].apply(preprocess_text)
    
    return df

def build_sentiment_dataset(df):
    rows = []
    for _, row in df.iterrows():
        text = row["review_text"]
        aspects = row["aspects"]
        sentiments = row["aspect_sentiments"]

        for aspect in aspects:
            rows.append({
                "text": text,
                "aspect": aspect,
                "sentiment": sentiments[aspect],
                "input_text": f"[ASPECT] {aspect} [TEXT] {text}"
            })
    
    out_df = pd.DataFrame(rows)
    out_df["label"] = out_df["sentiment"].map(SENTIMENT_LABELS)
    return out_df

class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

class SentimentDataset(Dataset):
    def __init__(self, input_texts, labels, tokenizer, max_len=128):
        self.input_texts = input_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        text = str(self.input_texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
