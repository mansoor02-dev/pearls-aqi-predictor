# Design Pattern: Schema validation with Pydantic
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationError

from src.utils.logger import setup_logger
from src.utils.exceptions import DataValidationError

logger = setup_logger(__name__)


class RawDataSchema(BaseModel):
    """Validates raw API response before feature engineering."""
    date: datetime
    city: str
    lat: float
    lon: float
    temperature_2m : Optional[float] = Field(..., ge=-273.15, le=60)
    relative_humidity_2m : Optional[float] = Field(..., ge=0)
    wind_speed_10m : Optional[float] = Field(..., ge=0)
    rain: Optional[float] = Field(None, ge=0)
    pm10: Optional[float] = Field(None, ge=0)
    pm2_5: Optional[float] = Field(None, ge=0)
    carbon_monoxide: Optional[float] = Field(None, ge=0)
    nitrogen_dioxide: Optional[float] = Field(None, ge=0)
    sulphur_dioxide: Optional[float] = Field(None, ge=0)
    ozone: Optional[float] = Field(None, ge=0)
    dust: Optional[float] = Field(None, ge=0)
    uv_index: Optional[float] = Field(None, ge=0)
    uv_index_clear_sky: Optional[float] = Field(None, ge=0)
    aerosol_optical_depth: Optional[float] = Field(None, ge=0)
    european_aqi: float = Field(..., ge=0, le=500)

    @field_validator("european_aqi")
    @classmethod
    def aqi_must_be_reasonable(cls, v):
        if v > 300:
            logger.warning(f"Extremely high AQI detected: {v}")
        return v


class DataValidator:
    """Validates DataFrames against schemas and business rules."""

    def validate_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        validation_errors = []
        for idx, row in df.iterrows():
            try:
                RawDataSchema(**row.to_dict())
            except DataValidationError as e:
                for err_dict in e.errors():
                    validation_errors.append({
                        **err_dict,
                        "loc": ("row", idx, *err_dict.get("loc", ())),
                        "input": row.to_dict() # Optionally add the input data that caused the error
                    })
        if validation_errors:
            raise DataValidationError(f"{len(validation_errors)} row(s) failed: {validation_errors[:3]}")
        
        null_rates = df.isnull().mean() * 100
        bad_cols = null_rates[null_rates > 5]
        if not bad_cols.empty:
            raise DataValidationError(f"Columns exceed 5% null rate: {bad_cols.to_dict()}")

        out_of_range = ~df["european_aqi"].between(0, 500)
        if out_of_range.any():
            raise DataValidationError(f"{out_of_range.sum()} row(s) with european_aqi out of [0, 500]")

        gaps = df["date"].diff() > pd.Timedelta(hours=3)
        if gaps.any():
            logger.warning(f"{gaps.sum()} timestamp gap(s) > 3h detected")

        return df

    def build_reference_stats(self, train_df: pd.DataFrame) -> dict:
        columns=train_df.select_dtypes(include=["number"]).columns
        return {
            col: {"mean": train_df[col].mean(), "std": train_df[col].std()}
            for col in columns
        }

    def validate_features(self, df: pd.DataFrame, reference_stats: dict = None) -> pd.DataFrame:
        """
        Checks engineered features for drift against a reference distribution.
        reference_stats: {column_name: {"mean": ..., "std": ...}}, built once
        from your training set and saved (e.g. as JSON) for reuse.
        """
        if reference_stats is None:
            logger.warning("No reference stats provided — skipping drift check.")
            return df

        for col, stats in reference_stats.items():
            if col not in df.columns:
                continue
            current_mean = df[col].mean()
            # z-score style check: how many std devs away is the new mean?
            if stats["std"] > 0:
                drift = abs(current_mean - stats["mean"]) / stats["std"]
                if drift > 3:
                    logger.warning(
                        f"Feature drift detected in '{col}': "
                        f"current mean={current_mean:.2f}, reference mean={stats['mean']:.2f}"
                    )
        return df