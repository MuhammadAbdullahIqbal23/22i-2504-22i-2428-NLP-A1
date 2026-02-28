"""
Stage 6 — Airflow Pipeline Orchestration
DAG: nlp_trend_pipeline
Tasks: scrape_data → preprocess_data → generate_features → compute_statistics → dvc_push
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

import sys
import os

# Add src/ to Python path so imports work inside Airflow
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# ── Default arguments ──
default_args = {
    "owner": "trendscope",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

# ── DAG definition ──
dag = DAG(
    dag_id="nlp_trend_pipeline",
    default_args=default_args,
    description="Reproducible NLP Trend Intelligence Pipeline",
    schedule_interval=None,          # Manually triggerable
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["nlp", "trendscope"],
)


# ── Task callables ──
def _scrape(**kwargs):
    import logging
    logger = logging.getLogger("airflow.task")
    logger.info("Starting data scrape …")
    from scraper import run as scrape_run
    products = scrape_run(target_count=300)
    logger.info("Scrape complete — %d products", len(products))


def _preprocess(**kwargs):
    import logging
    logger = logging.getLogger("airflow.task")
    logger.info("Starting preprocessing …")
    from preprocess import run as preprocess_run
    preprocess_run()
    logger.info("Preprocessing complete.")


def _features(**kwargs):
    import logging
    logger = logging.getLogger("airflow.task")
    logger.info("Generating feature representations …")
    from representation import run as repr_run
    repr_run()
    logger.info("Feature generation complete.")


def _statistics(**kwargs):
    import logging
    logger = logging.getLogger("airflow.task")
    logger.info("Computing linguistic statistics …")
    from statistics import run as stats_run
    stats_run()
    logger.info("Statistics report generated.")


# ── Tasks ──
scrape_data = PythonOperator(
    task_id="scrape_data",
    python_callable=_scrape,
    dag=dag,
)

preprocess_data = PythonOperator(
    task_id="preprocess_data",
    python_callable=_preprocess,
    dag=dag,
)

generate_features = PythonOperator(
    task_id="generate_features",
    python_callable=_features,
    dag=dag,
)

compute_statistics = PythonOperator(
    task_id="compute_statistics",
    python_callable=_statistics,
    dag=dag,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dvc_push = BashOperator(
    task_id="dvc_push",
    bash_command=f"cd {PROJECT_ROOT} && dvc add data/raw/products_raw.json data/processed/products_clean.csv data/features/ && dvc push",
    dag=dag,
)

# ── Dependencies ──
scrape_data >> preprocess_data >> generate_features >> compute_statistics >> dvc_push
