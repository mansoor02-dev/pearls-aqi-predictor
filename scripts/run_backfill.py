import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.backfill import backfill_historical_data

BACKFILL_START_DATE = "2023-01-01"

if __name__ == "__main__":
    end_date = datetime.now().strftime("%Y-%m-%d")
    backfill_historical_data(start_date=BACKFILL_START_DATE, end_date=end_date)