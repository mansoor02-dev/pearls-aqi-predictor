import pandas as pd
import numpy as np
from typing import Tuple

from src.models.base_model import BaseAQIModel

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



class LSTMAQIModel(BaseAQIModel):
    """
    LSTM for time series forecasting.
    Input shape: (timesteps, features)
    Output shape: (forecast_horizon,)
    """
    
    def __init__(self, model_name: str, sequence_length: int = 168):  # 7 days of hourly data
        super().__init__(model_name)
        self.sequence_length = sequence_length
        self.scaler = None  # You'll need MinMaxScaler
    
    def build_model(self, n_features: int) -> tf.keras.Model:
        model = Sequential([
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)  # Output: 3 days
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Create sequences: [t-168, ..., t-1] -> [t, t+1, t+2]
        # This is different from sklearn — needs sliding window
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = self.build_model(X_train.shape[2])
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return {'final_loss': history.history['loss'][-1]}