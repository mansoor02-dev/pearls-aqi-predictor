import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.backfill import backfill_historical_data

if __name__ == "__main__":
    backfill_historical_data(start_date="2023-01-01", end_date="2026-08-07")