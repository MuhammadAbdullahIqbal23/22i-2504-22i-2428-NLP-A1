"""
Stage 5 — Basic Linguistic Intelligence
Generates trend_summary.txt with unigram/bigram rankings, tag stats,
duplicate detection via Minimum Edit Distance.
"""

import csv
import json
import os
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "products_clean.csv")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "products_raw.json")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
REPORT_DIR = os.path.join(BASE_DIR, "reports")


# ── Minimum Edit Distance (Levenshtein) — manual implementation ──
def min_edit_distance(s1, s2):
    """Compute Levenshtein distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


# ── Unigram language model ──
def compute_unigram_probs(docs):
    """Return dict of token -> probability based on MLE."""
    counter = Counter(tok for doc in docs for tok in doc)
    total = sum(counter.values())
    probs = {tok: count / total for tok, count in counter.items()}
    return probs


def run(clean_path=None, raw_path=None):
    clean_path = clean_path or CLEAN_PATH
    raw_path = raw_path or RAW_PATH
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load cleaned data
    docs = []
    raw_texts = []
    with open(clean_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = row["tokens"].split("|") if row["tokens"] else []
            docs.append(tokens)
            raw_texts.append(row["text_raw"])

    # Load raw JSON for tags
    with open(raw_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    # ── Unigram & bigram frequencies ──
    unigram_counter = Counter(tok for doc in docs for tok in doc)
    bigram_counter = Counter()
    for doc in docs:
        for a, b in zip(doc, doc[1:]):
            bigram_counter[(a, b)] += 1

    top30_uni = unigram_counter.most_common(30)
    top20_bi = bigram_counter.most_common(20)

    # ── Tag / category stats ──
    tag_counter = Counter()
    for p in products:
        for tag in p.get("tags", []):
            tag_counter[tag] += 1
    top_tags = tag_counter.most_common(15)

    # ── Vocab size ──
    vocab = sorted({tok for doc in docs for tok in doc})
    vocab_size = len(vocab)

    # ── Average description length (in tokens) ──
    lengths = [len(doc) for doc in docs]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    # ── Duplicate detection via MED ──
    names = [p.get("product_name", "") for p in products]
    threshold = 3
    near_dupes = []
    # Pairwise comparison (cap to first 200 for speed)
    cap = min(len(names), 200)
    for i in range(cap):
        for j in range(i + 1, cap):
            d = min_edit_distance(names[i].lower(), names[j].lower())
            if 0 < d <= threshold:
                near_dupes.append((names[i], names[j], d))
    near_dupes.sort(key=lambda x: x[2])

    # ── Unigram probabilities ──
    probs = compute_unigram_probs(docs)

    # ── Write report ──
    report_path = os.path.join(REPORT_DIR, "trend_summary.txt")
    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 70 + "\n")
        rpt.write("  TrendScope Analytics — NLP Trend Intelligence Report\n")
        rpt.write("=" * 70 + "\n\n")

        rpt.write("TOP 30 UNIGRAMS\n")
        rpt.write("-" * 40 + "\n")
        for rank, (word, freq) in enumerate(top30_uni, 1):
            rpt.write(f"  {rank:>2}. {word:<25} {freq}\n")

        rpt.write("\nTOP 20 BIGRAMS\n")
        rpt.write("-" * 40 + "\n")
        for rank, ((a, b), freq) in enumerate(top20_bi, 1):
            rpt.write(f"  {rank:>2}. {a} {b:<30} {freq}\n")

        rpt.write("\nMOST COMMON TAGS / CATEGORIES\n")
        rpt.write("-" * 40 + "\n")
        for rank, (tag, freq) in enumerate(top_tags, 1):
            rpt.write(f"  {rank:>2}. {tag:<25} {freq}\n")

        rpt.write(f"\nVOCABULARY SIZE: {vocab_size}\n")
        rpt.write(f"AVERAGE DESCRIPTION LENGTH (tokens): {avg_len:.2f}\n")
        rpt.write(f"TOTAL DOCUMENTS: {len(docs)}\n")

        rpt.write("\nNEAR-DUPLICATE TITLES (Min Edit Distance ≤ {0})\n".format(threshold))
        rpt.write("-" * 40 + "\n")
        if near_dupes:
            for a, b, d in near_dupes[:20]:
                rpt.write(f"  \"{a}\" ↔ \"{b}\"  (dist={d})\n")
        else:
            rpt.write("  No near-duplicates detected.\n")

        rpt.write("\nUNIGRAM PROBABILITIES (top 15)\n")
        rpt.write("-" * 40 + "\n")
        for word, freq in top30_uni[:15]:
            rpt.write(f"  P({word}) = {probs[word]:.6f}\n")

    logger.info("Report saved → %s", report_path)
    return report_path


if __name__ == "__main__":
    run()
