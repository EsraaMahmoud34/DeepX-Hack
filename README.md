# 🧠 Arabic ABSA System — DeepX Hackathon 2026

Production-grade Arabic Aspect-Based Sentiment Analysis (ABSA) system built using a two-model AraBERT pipeline for extracting aspect-level sentiment from Arabic customer reviews.

---

# 🚀 About the Hackathon

This project was developed during **DeepX Hackathon 2025**.

The challenge focused on building intelligent AI systems capable of solving real-world Arabic NLP problems. Our team designed a complete Arabic NLP pipeline capable of understanding customer opinions at a much deeper level than traditional sentiment analysis systems.

Instead of predicting only one sentiment for the entire review, our system detects:
- the aspects mentioned in the review
- the sentiment for each aspect independently

This allows businesses to understand exactly what customers like or dislike.

---

# 📌 Project Summary

Traditional sentiment analysis assigns ONE sentiment score for the whole review.

Example:

> "الأكل ممتاز لكن الخدمة سيئة"

A traditional model may classify this review as neutral.

Our system instead predicts:

```json
{
  "aspects": ["food", "service"],
  "aspect_sentiments": {
    "food": "positive",
    "service": "negative"
  }
}
```

This provides much more actionable business insights.

---

# ✨ Features

✅ Arabic-specific preprocessing pipeline  
✅ Emoji-aware sentiment handling  
✅ Multi-label aspect detection  
✅ Aspect-conditioned sentiment classification  
✅ Cross-domain support  
✅ Production-style JSON output  
✅ AraBERT-based architecture  
✅ Handles Arabic dialects and noisy reviews  

---

# 🏗 System Architecture

```text
Raw Arabic Review
        ↓
Arabic Preprocessing
        ↓
Model 1 — Aspect Detection
        ↓
Model 2 — Sentiment Classification
        ↓
Structured JSON Output
```

---

# 🧹 Arabic Preprocessing Pipeline

The preprocessing pipeline was specially designed for noisy Arabic customer reviews.

It handles:
- Arabic normalization
- emojis
- elongation
- diacritics
- English-Arabic code-switching
- punctuation normalization
- business category injection
- star rating injection

---

## 🔧 Preprocessing Steps

### 1️⃣ Unicode Arabic Normalization
Normalizes Arabic letter variants:

```text
أ / إ / آ → ا
ى → ي
```

---

### 2️⃣ Emoji → Arabic Sentiment Mapping

Example:

```text
😍 → رائع جدا
👎 → سيء
❤ → محبوب
😡 → غضب شديد
```

This preserves sentiment information instead of removing emojis.

---

### 3️⃣ Elongation Normalization

Before:
```text
رائعةةةةة جداااا
```

After:
```text
رائعةة جداا
```

---

### 4️⃣ Whitespace & Newline Cleaning

Before:
```text
الاكل ممتاز

الخدمة سيئة
```

After:
```text
الاكل ممتاز الخدمة سيئة
```

---

### 5️⃣ English-Arabic Code-Switching Support

Before:
```text
The FOOD is Amazing والخدمة ممتازة
```

After:
```text
the food is amazing والخدمة ممتازة
```

---

### 6️⃣ Context Prefix Injection

The model receives additional business context:

```text
[مطعم] [نجوم:2]
```

This significantly improves understanding of ambiguous reviews.

---

## 🧪 Preprocessing Example

### Raw Input
```text
الأكل رائعةةة 😍😍 بس التوصيل اتأخر 👎
```

### Cleaned Output
```text
[مطعم] [نجوم:2] الاكل رائعةة رائع جدا بس التوصيل اتاخر سيء
```

---

# 🤖 Models

## Model 1 — Aspect Detection

Detects all aspects mentioned in the review using multi-label classification.

### Supported Aspects
- food
- service
- price
- cleanliness
- delivery
- ambiance
- app_experience
- general
- none

### Architecture
- AraBERT backbone
- Mean pooling
- BCEWithLogitsLoss
- Per-aspect threshold tuning

---

## Model 2 — Sentiment Classification

Predicts sentiment for each detected aspect independently.

### Sentiment Classes
- positive
- neutral
- negative

### Input Format
```text
[ASPECT] food [TEXT] الاكل ممتاز
```

### Architecture
- AraBERT backbone
- CLS pooling
- CrossEntropyLoss
- Inverse-frequency class weighting

---

# 📂 Dataset

The project uses:
- `train_fixed.xlsx`
- `validation_fixed.xlsx`
- `unlabeled_fixed.xlsx`

Dataset fields:
- `review_text`
- `aspects`
- `aspect_sentiments`
- `business_category`
- `star_rating`

---

# 📊 Training Strategy

## Training Setup
- Optimizer: AdamW
- Learning Rate: 2e-5
- Epochs: 20
- Batch Size: 16
- Max Length: 128

## Techniques Used
- Gradient clipping
- Linear warmup scheduler
- Threshold tuning
- Class imbalance handling

---

# 📁 Repository Structure

```text
.
├── app.py
├── train.py
├── inference.py
├── models.py
├── preprocessing.py
├── config.py
├── arabic_absa_system.ipynb
├── final notebook.ipynb
├── train_fixed.xlsx
├── validation_fixed.xlsx
├── unlabeled_fixed.xlsx
├── README.md
└── package-lock.json
```

---

# 🖼 Screenshots

## UI Example

Add your screenshots inside an `images/` folder.

Example:

```markdown
![UI Screenshot](<img width="1600" height="822" alt="image" src="https://github.com/user-attachments/assets/0891db56-f06d-47f0-9dc4-56fd5d1e6c6c" />
)
```

---

## Prediction Example

```markdown
![Prediction Example](<img width="1600" height="816" alt="image" src="https://github.com/user-attachments/assets/4742e7d1-368b-47be-af76-74af4cc0a2cb" />
)
(<img width="1600" height="903" alt="image" src="https://github.com/user-attachments/assets/b098aeae-0359-4665-8b31-b964153b5aee" />
)
(<img width="1600" height="757" alt="image" src="https://github.com/user-attachments/assets/31abdaea-c60e-4baf-9a71-0ade8a587aa4" />
)
(<img width="1600" height="856" alt="image" src="https://github.com/user-attachments/assets/b8929d60-933c-48ed-90c3-7bc124ff1a46" />
)
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/DeepX-Hack.git

cd DeepX-Hack
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Train Models

```bash
python train.py
```

---

## Run Inference

```bash
python inference.py
```

---

## Run UI

```bash
python app.py
```

---

# 💡 Example Prediction

## Input Review

```text
الأكل ممتاز لكن الخدمة بطيئة والأسعار غالية
```

## Output

```json
{
  "aspects": ["food", "service", "price"],
  "aspect_sentiments": {
    "food": "positive",
    "service": "negative",
    "price": "negative"
  }
}
```

---

# 🎯 Business Value

This system helps businesses:
- understand customer feedback in detail
- detect service problems early
- monitor delivery satisfaction
- analyze app experience
- improve customer retention

Applicable to:
- restaurants
- healthcare
- e-commerce
- delivery platforms
- retail businesses

---

# 🔮 Future Improvements

- REST API deployment
- Real-time dashboard
- More Arabic dialect support
- Larger Arabic datasets
- Model optimization for faster inference
- SaaS deployment

---

# 👥 Team

- :contentReference[oaicite:1]{index=1}
- :contentReference[oaicite:2]{index=2}
- :contentReference[oaicite:3]{index=3}

---

# 📜 License

MIT License

---

# ⭐ Acknowledgments

Special thanks to:
- DeepX Hackathon 2026
- AraBERT creators
- Open-source Arabic NLP community

---
