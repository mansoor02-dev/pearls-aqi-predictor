"""
src/utils/hopsworks_utils.py

Central authentication helper for Hopsworks (DRY).
"""
import hopsworks
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def login_hopsworks():
    """
    Centralized login function for Hopsworks.
    Reused across feature pipeline, backfill, training, inference, API, and dashboard.
    """
    logger.info(f"Connecting to Hopsworks project '{settings.HOPSWORKS_PROJECT_NAME}'...")
    return hopsworks.login(
        api_key_value=settings.HOPSWORKS_API_KEY,
        project=settings.HOPSWORKS_PROJECT_NAME,
        host=settings.HOPSWORKS_HOST,
    )
