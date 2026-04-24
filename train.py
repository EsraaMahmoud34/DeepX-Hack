import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report

from config import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, MODEL_NAME, MAX_LEN, 
    BATCH_SIZE, LEARNING_RATE, EPOCHS, ASPECTS, SENTIMENT_LABELS, OUTPUT_DIR
)
from dataset import load_data, build_sentiment_dataset, AspectDataset, SentimentDataset
from models import ABSAAspectModel, ABSASentimentModel

def train_aspect_model(train_loader, val_loader, tokenizer, device):
    print("--- Training Aspect Model ---")
    model = ABSAAspectModel(MODEL_NAME, len(ASPECTS)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")
        
        # Eval
        best_t, best_f1 = evaluate_aspect_model(model, val_loader, device)
        print(f"Validation Macro F1: {best_f1:.4f} at Threshold {best_t:.2f}")

    # Save
    import json
    os.makedirs(os.path.join(OUTPUT_DIR, "aspect_model"), exist_ok=True)
    model.model.save_pretrained(os.path.join(OUTPUT_DIR, "aspect_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "aspect_model"))
    
    with open(os.path.join(OUTPUT_DIR, "aspect_model", "threshold.json"), "w") as f:
        json.dump({"best_threshold": float(best_t)}, f)
        
    return model

def evaluate_aspect_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits, _ = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    best_t = 0.5
    best_f1 = 0
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

def train_sentiment_model(train_loader, val_loader, tokenizer, device):
    print("--- Training Sentiment Model ---")
    model = ABSASentimentModel(MODEL_NAME, len(SENTIMENT_LABELS)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")
        
        # Eval
        f1, acc = evaluate_sentiment_model(model, val_loader, device)
        print(f"Validation Macro F1: {f1:.4f}, Accuracy: {acc:.4f}")

    # Save
    os.makedirs(os.path.join(OUTPUT_DIR, "sentiment_model"), exist_ok=True)
    model.model.save_pretrained(os.path.join(OUTPUT_DIR, "sentiment_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "sentiment_model"))
    return model

def evaluate_sentiment_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load Data
    train_df = load_data(TRAIN_DATA_PATH)
    val_df = load_data(VAL_DATA_PATH)

    # 1. Aspect Data Loaders
    aspect_train_ds = AspectDataset(train_df["review_text"].values, np.stack(train_df["aspect_labels"].values), tokenizer, MAX_LEN)
    aspect_val_ds = AspectDataset(val_df["review_text"].values, np.stack(val_df["aspect_labels"].values), tokenizer, MAX_LEN)

    aspect_train_loader = DataLoader(aspect_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    aspect_val_loader = DataLoader(aspect_val_ds, batch_size=BATCH_SIZE)

    # Train Aspect Model
    train_aspect_model(aspect_train_loader, aspect_val_loader, tokenizer, device)

    # 2. Sentiment Data Loaders
    train_sent_df = build_sentiment_dataset(train_df)
    val_sent_df = build_sentiment_dataset(val_df)

    sent_train_ds = SentimentDataset(train_sent_df["input_text"].values, train_sent_df["label"].values, tokenizer, MAX_LEN)
    sent_val_ds = SentimentDataset(val_sent_df["input_text"].values, val_sent_df["label"].values, tokenizer, MAX_LEN)

    sent_train_loader = DataLoader(sent_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    sent_val_loader = DataLoader(sent_val_ds, batch_size=BATCH_SIZE)

    # Train Sentiment Model
    train_sentiment_model(sent_train_loader, sent_val_loader, tokenizer, device)

if __name__ == "__main__":
    main()
