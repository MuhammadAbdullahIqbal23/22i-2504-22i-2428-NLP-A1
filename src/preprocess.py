"""
Stage 3 — Text Processing & Representation
Cleans and tokenizes product text data for NLP analysis.
"""

import json
import os
import re
import csv
import unicodedata
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is present
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "products_raw.json")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "products_clean.csv")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ────── Preprocessing helpers ──────

def unicode_normalize(text):
    """NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)

def remove_html(text):
    """Strip HTML tags."""
    return re.sub(r"<[^>]+>", " ", text)

def remove_urls(text):
    """Remove http/https URLs."""
    return re.sub(r"https?://\S+", " ", text)

def remove_punctuation(text):
    """Remove all punctuation characters."""
    return re.sub(r"[^\w\s]", " ", text)

def remove_numeric_only(tokens):
    """Drop tokens that are purely numeric."""
    return [t for t in tokens if not t.isdigit()]

def remove_short_tokens(tokens, min_len=2):
    """Remove tokens shorter than min_len."""
    return [t for t in tokens if len(t) >= min_len]


def clean_text(text):
    """Full cleaning pipeline for a single text string."""
    text = unicode_normalize(text)
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    tokens = remove_numeric_only(tokens)
    tokens = remove_short_tokens(tokens)
    return tokens


def preprocess(raw_path=None, clean_path=None):
    raw_path = raw_path or RAW_PATH
    clean_path = clean_path or CLEAN_PATH

    with open(raw_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)

    rows = []
    for p in products:
        name = p.get("product_name", "")
        tagline = p.get("tagline", "")
        text_raw = f"{name} {tagline}".strip()
        tokens = clean_text(text_raw)
        text_clean = " ".join(tokens)
        rows.append({
            "text_raw": text_raw,
            "text_clean": text_clean,
            "tokens": "|".join(tokens),       # pipe-delimited for CSV safety
            "token_count": len(tokens),
        })

    with open(clean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_raw", "text_clean", "tokens", "token_count"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Cleaned %d records → %s", len(rows), clean_path)
    return clean_path


def run():
    return preprocess()


if __name__ == "__main__":
    run()
