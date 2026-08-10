"""
src/pipeline/feature_pipeline.py

Scheduled feature pipeline update.
Calculates lookback date range and delegates to backfill_historical_data.
"""
from datetime import datetime, timedelta

from src.features.backfill import backfill_historical_data
from src.utils.logger import setup_logger

logger = setup_logger("FeaturePipeline")
LOOKBACK_DAYS = 16


def update_features(lookback_days: int = LOOKBACK_DAYS):
    """
    Run hourly feature pipeline update for the past `lookback_days`.
    """
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days + 1)).strftime("%Y-%m-%d")

    logger.info(f"Starting feature pipeline update: {start_date} to {end_date}")
    return backfill_historical_data(start_date=start_date, end_date=end_date)
