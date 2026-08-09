import hopsworks
from hopsworks.client.exceptions import RestAPIError
import pandas as pd
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)


class HopsworksFeatureStore:
    """
    Wrapper around Hopsworks Feature Store.
    Handles connection, feature group creation, and materialization.

    Takes an already-authenticated `project` handle (from a single
    hopsworks.login() call, shared with HopsworksModelRegistry) rather than
    logging in itself — avoids a second, redundant session per pipeline run.
    """

    def __init__(self, project):
        self.project = project
        self.fs = project.get_feature_store()
        self.logger = setup_logger(self.__class__.__name__)

    def create_or_get_feature_group(self, name: str, version: int, primary_key: list, event_time: str):
        """Get the feature group at this EXACT version, or create it if it
        doesn't exist yet. Different hopsworks SDK versions handle "not found"
        differently — some raise RestAPIError, some just return None. Handle
        both rather than assume one."""
        try:
            fg = self.fs.get_feature_group(name=name, version=version)
        except RestAPIError:
            fg = None

        if fg is not None:
            self.logger.info(f"Found existing feature group: {name}_v{version}")
            return fg

        fg = self.fs.create_feature_group(
            name=name,
            version=version,
            description="AQI and weather features for forecasting",
            primary_key=primary_key,
            event_time=event_time,
            online_enabled=True,
        )
        self.logger.info(f"Created feature group: {name}_v{version}")
        return fg

    def list_feature_group_versions(self, name: str) -> list:
        versions = [fg.version for fg in self.fs.get_feature_groups(name)]
        self.logger.info(f"Existing versions of '{name}': {sorted(versions)}")
        return sorted(versions)

    def append_features(self, feature_group, new_features: list):
        feature_group.append_features(new_features)
        self.logger.info(f"Appended {len(new_features)} feature(s) to {feature_group.name}_v{feature_group.version}")

    def create_new_feature_group_version(self, name: str, primary_key: list, event_time: str, description: str = ""):
        fg = self.fs.create_feature_group(
            name=name, version=None, description=description,
            primary_key=primary_key, event_time=event_time, online_enabled=True,
        )
        self.logger.info(f"Created NEW schema version: {name}_v{fg.version} — update FEATURE_GROUP_VERSION in settings.py")
        return fg

    def insert_features(self, feature_group, df: pd.DataFrame):
        df = df.copy()    
        df["date"] = pd.to_datetime(df["date"])
        
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        
        df["date"] = df["date"].astype("datetime64[us]") 
        
        feature_group.insert(df, write_options={"wait_for_job": True})
        self.logger.info(f"Inserted {len(df)} rows into feature group")

    def get_training_data(self, feature_view_name: str, start_date: str = None, end_date: str = None):    
        """Creates a feature view, materializes a versioned
        training dataset for the given time range, and returns it as a
        DataFrame + labels. Each call with a distinct date range creates its
        own training-dataset version"""
        query = self.fs.get_feature_group(
            settings.FEATURE_GROUP_NAME, version=settings.FEATURE_GROUP_VERSION
        ).select_all()

        try:
            fv = self.fs.get_feature_view(name=feature_view_name, version=settings.FEATURE_VIEW_VERSION)
        except RestAPIError:
            fv = None

        if fv is not None:
            self.logger.info(f"Found existing feature view: {feature_view_name}_v{settings.FEATURE_VIEW_VERSION}")
        else:
            fv = self.fs.create_feature_view(
                name=feature_view_name,
                version=settings.FEATURE_VIEW_VERSION,
                query=query,
                labels=[f"aqi_next_{d}d" for d in range(1, settings.FORECAST_HORIZON + 1)],
            )
            self.logger.info(f"Created feature view: {feature_view_name}_v{settings.FEATURE_VIEW_VERSION}")

        td_version, _job = fv.create_training_data(
            start_time=start_date, end_time=end_date,
            description=f"Training data {start_date} to {end_date}",
        )
        feature_df, label_df = fv.get_training_data(training_dataset_version=td_version)
        return pd.concat([feature_df, label_df], axis=1)