# 🧠 Arabic ABSA System — DeepX Hackathon 2026

Production-grade Arabic Aspect-Based Sentiment Analysis (ABSA) system built using a two-model AraBERT pipeline for extracting aspect-level sentiment from Arabic customer reviews.

---

# 🚀 About the Hackathon

This project was developed during **DeepX Hackathon 2026**.

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

# 🖼 System Architecture Diagram

PASTE ARCHITECTURE IMAGE HERE

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

```text
أ / إ / آ → ا
ى → ي
```

---

### 2️⃣ Emoji → Arabic Sentiment Mapping

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

# 🖼 Preprocessing Example Screenshot

PASTE PREPROCESSING IMAGE HERE

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

# 🏆 Hackathon Results

Our team achieved strong performance during DeepX Hackathon 2026:

- 🥇 Ranked **15th out of 150 teams**
- 📊 Achieved an **F1-score of 0.8425** on the training set
- 🎯 Submission score: **~24ز94/30**

This project successfully competed among top-performing Arabic NLP solutions in the hackathon.

---

# 🖼 Leaderboard & Score Screenshot

<img width="1600" height="722" alt="52c2a9b2-cabc-4871-a3c3-fef166746555" src="https://github.com/user-attachments/assets/c8a4c16c-68e3-42c0-8bf3-cbf3e74fa9a8" />

<img width="1600" height="308" alt="163aa547-f323-4449-bc2e-15c86b98408d" src="https://github.com/user-attachments/assets/de5b24ca-111d-4074-9191-02143890397b" />


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

# 🖼 UI Demo


<img width="1600" height="822" alt="d78908f7-fb50-46f2-8629-098adbea7004" src="https://github.com/user-attachments/assets/f80b27c4-ba44-4683-8128-dd7e63602d2c" />

---

# 🖼 Prediction Examples

PREDICTION IMAGE 1
<img width="1600" height="816" alt="05f4e7e2-abe5-4892-ab90-64a3a562e4ba" src="https://github.com/user-attachments/assets/7ccd18c4-7799-46d8-94be-a1e06cdc2c59" />

<br>

PREDICTION IMAGE 2 
<img width="1600" height="903" alt="9fbc9ac2-0533-4da7-b7f7-0e695807f37a" src="https://github.com/user-attachments/assets/87790401-ee6b-40fd-9793-dcc0968ba108" />

<br>

PREDICTION IMAGE 3
<img width="1600" height="757" alt="8c80ef17-9d73-4b84-95a0-923ad5078c30" src="https://github.com/user-attachments/assets/243b9696-f96b-40fa-ba6a-c09ef38a4ed1" />

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/EsraaMahmoud34/DeepX-Hack.git

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

- Esraa Mahmoud
- Ahmed Talima
- Yousef Samir

---

# 📜 License

MIT License

---

# ⭐ Acknowledgments

Special thanks to:
- DeepX Hackathon 2026
- AraBERT creators
- Open-source Arabic NLP community
