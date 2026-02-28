"""
Stage 4 — Data Representation
Vocabulary extraction, One-Hot Encoding, Bag-of-Words, N-gram frequencies.
"""

import csv
import json
import os
import logging
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "products_clean.csv")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")


def _load_tokens(clean_path=None):
    clean_path = clean_path or CLEAN_PATH
    docs = []
    with open(clean_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = row["tokens"].split("|") if row["tokens"] else []
            docs.append(tokens)
    return docs


# ── Vocabulary extraction ──
def build_vocab(docs):
    vocab = sorted({tok for doc in docs for tok in doc})
    logger.info("Vocabulary size: %d", len(vocab))
    return vocab


# ── One-Hot Encoding (small subset — first 20 docs) ──
def one_hot_encode(docs, vocab, subset_size=20):
    word2idx = {w: i for i, w in enumerate(vocab)}
    subset = docs[:subset_size]
    matrix = np.zeros((len(subset), len(vocab)), dtype=np.int8)
    for i, doc in enumerate(subset):
        for tok in set(doc):
            if tok in word2idx:
                matrix[i, word2idx[tok]] = 1
    logger.info("One-Hot matrix shape: %s", matrix.shape)
    return matrix


# ── Bag-of-Words matrix ──
def bow_matrix(docs, vocab):
    word2idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(docs), len(vocab)), dtype=np.int32)
    for i, doc in enumerate(docs):
        for tok in doc:
            if tok in word2idx:
                mat[i, word2idx[tok]] += 1
    logger.info("BoW matrix shape: %s", mat.shape)
    return mat


# ── Unigram frequency ──
def unigram_freq(docs):
    counter = Counter(tok for doc in docs for tok in doc)
    return counter


# ── Bigram frequency ──
def bigram_freq(docs):
    counter = Counter()
    for doc in docs:
        for a, b in zip(doc, doc[1:]):
            counter[(a, b)] += 1
    return counter


def run(clean_path=None):
    clean_path = clean_path or CLEAN_PATH
    os.makedirs(FEATURES_DIR, exist_ok=True)

    docs = _load_tokens(clean_path)
    vocab = build_vocab(docs)

    # Save vocab
    vocab_path = os.path.join(FEATURES_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    logger.info("Vocab saved → %s", vocab_path)

    # One-hot
    ohe = one_hot_encode(docs, vocab)
    ohe_path = os.path.join(FEATURES_DIR, "onehot_matrix.npy")
    np.save(ohe_path, ohe)

    # BoW
    bow = bow_matrix(docs, vocab)
    bow_path = os.path.join(FEATURES_DIR, "bow_matrix.npy")
    np.save(bow_path, bow)
    logger.info("BoW matrix saved → %s", bow_path)

    # Unigram freq
    uf = unigram_freq(docs)
    uf_path = os.path.join(FEATURES_DIR, "unigram_freq.json")
    with open(uf_path, "w", encoding="utf-8") as f:
        json.dump(uf.most_common(), f, indent=2)

    # Bigram freq
    bf = bigram_freq(docs)
    bf_sorted = bf.most_common()
    bf_path = os.path.join(FEATURES_DIR, "bigram_freq.json")
    with open(bf_path, "w", encoding="utf-8") as f:
        json.dump([(" ".join(k), v) for k, v in bf_sorted], f, indent=2)

    logger.info("Feature generation complete.")
    return vocab, bow, uf, bf


if __name__ == "__main__":
    run()
