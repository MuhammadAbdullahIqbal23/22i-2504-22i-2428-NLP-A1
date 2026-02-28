# TrendScope Analytics — NLP Trend Intelligence Pipeline

A reproducible NLP data pipeline that collects, processes, represents, versions, and analyzes product listing data from Product Hunt.

## Quick Start

```bash
pip install -r requirements.txt
python src/scraper.py          # Stage 1: collect 300+ products
python src/preprocess.py       # Stage 3: clean & tokenize
python src/representation.py   # Stage 4: BoW, One-Hot, N-grams
python src/statistics.py       # Stage 5: generate trend report
```

## DVC Setup

```bash
dvc init
dvc remote add -d dagshub https://dagshub.com/<user>/<repo>.dvc
dvc add data/raw/products_raw.json
dvc push
```

## Airflow

Copy `dags/nlp_trend_dag.py` into your Airflow DAGs folder and trigger manually.
