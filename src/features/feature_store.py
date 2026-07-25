import hopsworks
from hopsworks.client.exceptions import DatasetException
import pandas as pd

from src.utils.logger import setup_logger
from config.settings import settings

class HopsworksFeatureStore:
    """
    Wrapper around Hopsworks Feature Store.
    Handles connection, feature group creation, and materialization.
    """
    
    def __init__(self, api_key: str, project_name: str, host: str, port: int = 443):
        self.project = hopsworks.login(
            api_key_value=api_key, 
            project=project_name,
            host=host,
            port=port
        )
        
        self.fs = self.project.get_feature_store()
        
        self.logger = setup_logger(self.__class__.__name__)
    
    def create_or_get_feature_group(
        self,
        name: str,
        version: int,
        primary_key: list,
        event_time: str
    ):
        """
        Create feature group if it doesn't exist.
        Primary key: ['city', 'timestamp']
        Event time: 'timestamp' (for time-travel queries)
        """
        try:
            fg = self.fs.get_feature_group(name=name, version=version)
            self.logger.info(f"Found existing feature group: {name}_v{version}")
            return fg
        except:
            # Create new feature group
            fg = self.fs.create_feature_group(
                name=name,
                version=version,
                description="AQI and weather features for forecasting",
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=True  # CRITICAL for real-time serving
            )
            self.logger.info(f"Created feature group: {name}_v{version}")
            return fg
    
    def insert_features(self, feature_group, df: pd.DataFrame):
        """Insert new rows into feature group."""
        feature_group.insert(df, write_options={"wait_for_job": True})
        self.logger.info(f"Inserted {len(df)} rows into feature group")
    
    def get_training_data(
        self,
        feature_view_name: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Create feature view and return training data.
        Feature view = query definition over feature groups.
        """
        # 1. Create query joining feature groups (if multiple)
        query = self.fs.get_feature_group(
            settings.FEATURE_GROUP_NAME, 
            version=settings.FEATURE_GROUP_VERSION
        ).select_all()
        
        # 2. Create or get feature view
        try:
            fv = self.fs.get_feature_view(name=feature_view_name, version=settings.FEATURE_VIEW_VERSION)
            self.logger.info(f"Found existing feature view: {feature_view_name}_v{settings.FEATURE_VIEW_VERSION}")
       
        except:
            fv = self.fs.create_feature_view(
                name=feature_view_name,
                version=settings.FEATURE_VIEW_VERSION,
                query=query,
                labels=[f"aqi_next_{d}d" for d in range(settings.FORECAST_HORIZON)]
            )
        self.logger.info(f"Created feature view: {feature_view_name}_v{settings.FEATURE_VIEW_VERSION}")
        # 3. Get training data with time split
        return fv.get_training_data(start_time=start_date, end_time=end_date)