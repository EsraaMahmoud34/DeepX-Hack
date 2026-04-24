import os
import json
import torch
from transformers import AutoTokenizer

from config import ASPECTS, ID_TO_SENTIMENT, OUTPUT_DIR, TEST_DATA_PATH, MAX_LEN
from preprocessing import preprocess_text
from models import ABSAAspectModel, ABSASentimentModel
from dataset import load_data

class ABSAPipeline:
    def __init__(self, aspect_model_dir, sentiment_model_dir, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load best threshold if exists
        threshold_path = os.path.join(aspect_model_dir, "threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                self.threshold = json.load(f).get("best_threshold", threshold)
        else:
            self.threshold = threshold

        print(f"Loading Tokenizer from {aspect_model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(aspect_model_dir)

        print(f"Loading Aspect Model from {aspect_model_dir}...")
        self.aspect_model = ABSAAspectModel(aspect_model_dir, len(ASPECTS)).to(self.device)
        self.aspect_model.eval()

        print(f"Loading Sentiment Model from {sentiment_model_dir}...")
        self.sentiment_model = ABSASentimentModel(sentiment_model_dir, len(ID_TO_SENTIMENT)).to(self.device)
        self.sentiment_model.eval()

    def predict(self, review_text, review_id=None):
        # Step 1: Preprocess
        clean_text = preprocess_text(review_text)

        # Step 2: Predict Aspects
        encoding = self.tokenizer(
            clean_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            aspect_logits, _ = self.aspect_model(input_ids, attention_mask)
            aspect_probs = torch.sigmoid(aspect_logits).cpu().numpy()[0]

        predicted_aspects = []
        for idx, prob in enumerate(aspect_probs):
            if prob > self.threshold:
                predicted_aspects.append(ASPECTS[idx])

        # If no aspect is predicted, fallback to "none" or "general"
        if len(predicted_aspects) == 0:
            predicted_aspects = ["none"]

        # Step 3: Predict Sentiment for each aspect
        aspect_sentiments = {}
        for aspect in predicted_aspects:
            input_text = f"[ASPECT] {aspect} [TEXT] {clean_text}"
            enc = self.tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            i_ids = enc["input_ids"].to(self.device)
            a_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                sent_logits, _ = self.sentiment_model(i_ids, a_mask)
                pred_idx = torch.argmax(sent_logits, dim=1).item()
                aspect_sentiments[aspect] = ID_TO_SENTIMENT[pred_idx]

        # Step 4: Build JSON output
        result = {
            "aspects": predicted_aspects,
            "aspect_sentiments": aspect_sentiments
        }
        if review_id is not None:
            result = {"review_id": review_id, **result}

        return result

def main():
    aspect_model_dir = os.path.join(OUTPUT_DIR, "aspect_model")
    sentiment_model_dir = os.path.join(OUTPUT_DIR, "sentiment_model")

    if not os.path.exists(aspect_model_dir) or not os.path.exists(sentiment_model_dir):
        print("Models not found. Please run train.py first.")
        return

    pipeline = ABSAPipeline(aspect_model_dir, sentiment_model_dir)

    # Test on a single sample
    sample_text = "الأكل ممتاز لكن الخدمة بطيئة والأسعار غالية"
    print("\nSample Inference:")
    print(json.dumps(pipeline.predict(sample_text, review_id=1), ensure_ascii=False, indent=2))

    # Process unlabeled dataset
    print(f"\nProcessing unlabeled dataset: {TEST_DATA_PATH}")
    if os.path.exists(TEST_DATA_PATH):
        df = load_data(TEST_DATA_PATH, is_inference=True)
        results = []

        for _, row in df.iterrows():
            review_id = row.get("review_id", None)
            if review_id is not None:
                review_id = int(review_id)
            res = pipeline.predict(row["review_text"], review_id=review_id)
            results.append(res)

        print("Saving sample output to outputs/submission_sample.json...")
        with open(os.path.join(OUTPUT_DIR, "submission_sample.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("Sample JSON Output:")
        print(json.dumps(results[:2], ensure_ascii=False, indent=2))
    else:
        print("Unlabeled test data not found.")

if __name__ == "__main__":
    main()
