import re

# -------------------------
# Emoji mapping (important for sentiment)
# -------------------------
emoji_map = {
    "❤️": " حب ",
    "😍": " حب ",
    "😂": " ضحك ",
    "😡": " غضب ",
    "👍": " جيد ",
    "👎": " سيء "
}

# -------------------------
# Remove elongation (هههههه → هه)
# -------------------------
def remove_elongation(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

# -------------------------
# Normalize Arabic
# -------------------------
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text

# -------------------------
# Remove diacritics
# -------------------------
def remove_diacritics(text):
    arabic_diacritics = re.compile("""
        ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ
    """, re.VERBOSE)
    return re.sub(arabic_diacritics, '', text)

# -------------------------
# Replace emojis with meaning
# -------------------------
def replace_emojis(text):
    for emoji, word in emoji_map.items():
        text = text.replace(emoji, word)
    return text

# -------------------------
# Remove URLs and noise
# -------------------------
def remove_noise(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------
# FULL PIPELINE
# -------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = replace_emojis(text)
    text = remove_elongation(text)
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_noise(text)

    return text
