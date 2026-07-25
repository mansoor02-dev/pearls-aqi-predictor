# Design Pattern: Strategy + Abstract Base Class
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from src.utils.exceptions import APIClientError
from src.utils.logger import setup_logger

AQI_VARIABLES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust", "uv_index",
    "uv_index_clear_sky", "aerosol_optical_depth", "european_aqi",
]
WEATHER_VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "rain"]


class BaseAPIClient(ABC):
    """Abstract base for all API clients."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def fetch_current(self, url: str, city: str, lat: float, lon: float) -> Dict[str, Any]:
        pass

    @abstractmethod
    def fetch_historical(self, url: str, city: str, lat: float, lon: float,
                          start_date: str, end_date: str) -> list:
        pass


class OpenMeteoClient(BaseAPIClient):
    """Open-Meteo AQI + Weather client (no API key needed)."""

    AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        super().__init__()
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    # ---- shared internals ----------------------------------------------

    def _fetch_current(self, url: str, variables: List[str], city: str,
                        lat: float, lon: float, prefix: str = "") -> Dict[str, Any]:
        params = {"latitude": lat, "longitude": lon, "current": variables, "timezone": "auto"}
        self.logger.info(f"Fetching current data from {url} for {city} ({lat}, {lon})")
        try:
            response = self.client.weather_api(url, params=params)[0]
            current = response.Current()
            data = {
                "date": pd.to_datetime(current.Time(), unit="s", utc=True)
                    .tz_convert(response.Timezone().decode()).tz_localize(None),
                "city": city, "lat": lat, "lon": lon,
            }
            data.update({
                f"{prefix}{var}": current.Variables(i).Value()
                for i, var in enumerate(variables)
            })
            return data
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise APIClientError(f"OpenMeteo current request failed ({url}): {e}")

    def _fetch_historical(self, url: str, variables: List[str], city: str,
                           lat: float, lon: float, start_date: str, end_date: str) -> list:
        params = {
            "latitude": lat, "longitude": lon, "hourly": variables,
            "timezone": "auto", "start_date": start_date, "end_date": end_date,
        }
        self.logger.info(f"Fetching historical data from {url} for {city} [{start_date} -> {end_date}]")
        try:
            response = self.client.weather_api(url, params=params)[0]
            hourly = response.Hourly()
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                ).tz_convert(response.Timezone().decode()),
                "city": city, "lat": lat, "lon": lon,
            }
            for i, var in enumerate(variables):
                hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
            return pd.DataFrame(hourly_data).to_dict(orient="records")
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise APIClientError(f"OpenMeteo historical request failed ({url}): {e}")

    # ---- public API -------------------------------------------------------

    def fetch_current(self, city: str, lat: float, lon: float) -> Dict[str, Any]:
        return self._fetch_current(self.AQI_URL, AQI_VARIABLES, city, lat, lon, prefix="current_")

    def fetch_historical(self, city: str, lat: float, lon: float,
                          start_date: str, end_date: str) -> list:
        return self._fetch_historical(self.AQI_URL, AQI_VARIABLES, city, lat, lon, start_date, end_date)

    def fetch_current_weather(self, city: str, lat: float, lon: float) -> Dict[str, Any]:
        return self._fetch_current(self.WEATHER_URL, WEATHER_VARIABLES, city, lat, lon)

    def fetch_historical_weather(self, city: str, lat: float, lon: float,
                              start_date: str, end_date: str) -> list:
        return self._fetch_historical(self.WEATHER_URL, WEATHER_VARIABLES, city, lat, lon, start_date, end_date)

class APIClientFactory:
    """Factory to get the active client. Only OpenMeteo is wired up."""

    @staticmethod
    def get_primary_client() -> BaseAPIClient:
        return OpenMeteoClient()