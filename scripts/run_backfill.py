import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.backfill import backfill_historical_data
from config.settings import settings


def main():
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    backfill_historical_data(start_date=settings.BACKFILL_START_DATE, end_date=end_date)


if __name__ == "__main__":
    main()