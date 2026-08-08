# src/models/deeplearning_model.py
import numpy as np
import pandas as pd
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

from src.models.base_model import BaseAQIModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEVICE = torch.device("cpu")   # training runs in GitHub Actions — no GPU, don't probe for cuda
BASE_DROP_COLS = ["date", "city", "lat", "lon"]


def _prepare_flat_split(df: pd.DataFrame, horizon: int, test_frac: float):
    """Shared by both models below — builds the flat (non-windowed) train/test
    split from the feature-engineered dataframe. Windowing (LSTM only) happens
    on top of this, after scaling, so it never crosses the train/test boundary."""
    target_col = f"aqi_next_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found — was AQIFeatureEngineer built "
                          f"with forecast_horizon >= {horizon}?")

    d = df.dropna(subset=[target_col]).copy()
    d = d.dropna()
    d["target_delta"] = d[target_col] - d["european_aqi"]

    other_horizon_cols = [c for c in d.columns if c.startswith("aqi_next_") and c != target_col]
    drop_cols = BASE_DROP_COLS + [target_col, "target_delta"] + other_horizon_cols

    X = d.drop(columns=drop_cols)
    y_delta = d["target_delta"].to_numpy()
    y_raw = d[target_col].to_numpy()
    current_aqi = d["european_aqi"].to_numpy()

    split_idx = int(len(X) * (1 - test_frac))
    return {
        "feature_names": X.columns.tolist(),
        "X_train": X.iloc[:split_idx].to_numpy(), "X_test": X.iloc[split_idx:].to_numpy(),
        "y_train_delta": y_delta[:split_idx], "y_test_delta": y_delta[split_idx:],
        "y_train_raw": y_raw[:split_idx], "y_test_raw": y_raw[split_idx:],
        "current_aqi_train": current_aqi[:split_idx], "current_aqi_test": current_aqi[split_idx:],
    }


def _make_windows(arr_2d: np.ndarray, sequence_length: int) -> np.ndarray:
    """(n, features) -> (n - sequence_length + 1, sequence_length, features)."""
    return np.array([arr_2d[i:i + sequence_length] for i in range(len(arr_2d) - sequence_length + 1)])


def _window_end_indices(n: int, sequence_length: int) -> np.ndarray:
    """Index of each window's LAST row — that row is "now"; targets/current_aqi
    must be read from this same index, not one step past the window."""
    return np.arange(sequence_length - 1, n)


def _train_torch_model(model: nn.Module, train_loader: DataLoader, epochs: int = 150,
                        lr: float = 1e-3, patience: int = 10) -> list:
    """Mini-batch training with early stopping, shared by both model classes below.
    Returns the per-epoch loss history for diagnostics plotting."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    best_loss, patience_left, best_state = float("inf"), patience, None
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= n_batches
        loss_history.append(epoch_loss)

        if epoch_loss < best_loss - 1e-4:
            best_loss, patience_left = epoch_loss, patience
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        if patience_left == 0:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return loss_history


def _score(model_name: str, forecast_horizon: int, y_true_raw, pred_raw, y_naive=None) -> Dict[str, float]:
    r2 = r2_score(y_true_raw, pred_raw)
    mae = mean_absolute_error(y_true_raw, pred_raw)
    mse = mean_squared_error(y_true_raw, pred_raw)
    rmse = root_mean_squared_error(y_true_raw, pred_raw)

    skill = None
    if y_naive is not None:
        naive_rmse = root_mean_squared_error(y_true_raw, y_naive)
        skill = 1 - (rmse / naive_rmse) if naive_rmse > 0 else np.nan

    metrics = {"model": model_name, "horizon": f"{forecast_horizon}d",
               "r2": r2, "mae": mae, "mse": mse, "rmse": rmse, "skill_vs_naive": skill}
    skill_str = f"   Skill: {skill:+.3f}" if skill is not None else ""
    logger.info(f"{model_name:32s}  R2: {r2:.4f}   MAE: {mae:.3f}   RMSE: {rmse:.3f}{skill_str}")
    return metrics


# ------------------------
# Feed-Forward NN
# ------------------------

class _FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(output_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.out(x)   # (batch, 1) — must match target shape exactly


class FeedForwardAQIModel(BaseAQIModel):
    def __init__(self, model_name: str, forecast_horizon: int = 1):
        super().__init__(model_name, forecast_horizon=forecast_horizon)
        self.x_scaler = RobustScaler()
        self.y_scaler = StandardScaler()
        self.loss_history_ = None
        self.feature_names_ = None

    def preprocess(self, df: pd.DataFrame, horizon: int = None, test_frac: float = 0.2) -> dict:
        horizon = horizon or self.forecast_horizon
        split = _prepare_flat_split(df, horizon, test_frac)
        self.feature_names_ = split["feature_names"]

        split["X_train"] = self.x_scaler.fit_transform(split["X_train"])
        split["X_test"] = self.x_scaler.transform(split["X_test"])
        split["y_train_delta"] = self.y_scaler.fit_transform(split["y_train_delta"].reshape(-1, 1)).ravel()
        # y_test_delta intentionally left unscaled
        return split

    def train(self, X_train, y_train_delta, X_val=None, y_val_delta=None, epochs=150, lr=1e-3) -> Dict[str, float]:
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train_delta, dtype=torch.float32).unsqueeze(1)   # (N,1) — matches model output shape
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)

        self.model = _FeedForwardNet(input_size=X_train.shape[1])
        self.loss_history_ = _train_torch_model(self.model, loader, epochs=epochs, lr=lr)
        self.is_trained = True
        return {"final_loss": self.loss_history_[-1]}

    def predict(self, X, current_aqi) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()
        pred_delta = self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        current_aqi = current_aqi.values if hasattr(current_aqi, "values") else np.asarray(current_aqi)
        return current_aqi + pred_delta

    def evaluate(self, X_test, y_test_raw, current_aqi_test, y_naive=None) -> Dict[str, float]:
        pred_raw = self.predict(X_test, current_aqi_test)
        self.metrics = _score(self.model_name, self.forecast_horizon, y_test_raw, pred_raw, y_naive)
        return self.metrics

    def get_feature_importance(self) -> Dict[str, float]:
        return {}   # no native importance measure — see self.loss_history_ for the NN-equivalent diagnostic

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "input_size": len(self.feature_names_),
                    "x_scaler": self.x_scaler, "y_scaler": self.y_scaler,
                    "forecast_horizon": self.forecast_horizon, "feature_names_": self.feature_names_}, path)
        logger.info(f"Saved {self.model_name} to {path}")

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model = _FeedForwardNet(input_size=state["input_size"])
        self.model.load_state_dict(state["state_dict"])
        self.x_scaler, self.y_scaler = state["x_scaler"], state["y_scaler"]
        self.forecast_horizon = state["forecast_horizon"]
        self.feature_names_ = state["feature_names_"]
        self.is_trained = True
        logger.info(f"Loaded {self.model_name} from {path}")


# --------------------------
# LSTM - sliding windows
# --------------------------

class _LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)                        # x: (batch, seq_len, features)
        out = self.dropout(out[:, -1, :])             # last timestep = "now"
        return self.fc(out)                           # (batch, 1) — matches target shape


class LSTMAQIModel(BaseAQIModel):
    def __init__(self, model_name: str, forecast_horizon: int = 1, sequence_length: int = 24):
        super().__init__(model_name, forecast_horizon=forecast_horizon)
        self.sequence_length = sequence_length
        self.x_scaler = RobustScaler()
        self.y_scaler = StandardScaler()
        self.loss_history_ = None
        self.feature_names_ = None

    def preprocess(self, df: pd.DataFrame, horizon: int = None, test_frac: float = 0.2) -> dict:
        horizon = horizon or self.forecast_horizon
        flat = _prepare_flat_split(df, horizon, test_frac)
        self.feature_names_ = flat["feature_names"]

        X_train_scaled = self.x_scaler.fit_transform(flat["X_train"])
        X_test_scaled = self.x_scaler.transform(flat["X_test"])
        y_train_delta_scaled = self.y_scaler.fit_transform(flat["y_train_delta"].reshape(-1, 1)).ravel()

        X_train_seq = _make_windows(X_train_scaled, self.sequence_length)
        X_test_seq = _make_windows(X_test_scaled, self.sequence_length)

        train_end_idx = _window_end_indices(len(X_train_scaled), self.sequence_length)
        test_end_idx = _window_end_indices(len(X_test_scaled), self.sequence_length)

        return {
            "X_train": X_train_seq, "X_test": X_test_seq,
            "y_train_delta": y_train_delta_scaled[train_end_idx],
            "y_test_delta": flat["y_train_delta"][train_end_idx] if False else flat["y_test_delta"][test_end_idx],
            "y_test_raw": flat["y_test_raw"][test_end_idx],
            "current_aqi_test": flat["current_aqi_test"][test_end_idx],
        }

    def train(self, X_train, y_train_delta, X_val=None, y_val_delta=None, epochs=100, lr=1e-3) -> Dict[str, float]:
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train_delta, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)

        self.model = _LSTMNet(input_size=X_train.shape[2])
        self.loss_history_ = _train_torch_model(self.model, loader, epochs=epochs, lr=lr)
        self.is_trained = True
        return {"final_loss": self.loss_history_[-1]}

    def predict(self, X, current_aqi) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()
        pred_delta = self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        current_aqi = current_aqi.values if hasattr(current_aqi, "values") else np.asarray(current_aqi)
        return current_aqi + pred_delta

    def evaluate(self, X_test, y_test_raw, current_aqi_test, y_naive=None) -> Dict[str, float]:
        pred_raw = self.predict(X_test, current_aqi_test)
        self.metrics = _score(self.model_name, self.forecast_horizon, y_test_raw, pred_raw, y_naive)
        return self.metrics

    def get_feature_importance(self) -> Dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "input_size": len(self.feature_names_),
                    "sequence_length": self.sequence_length, "x_scaler": self.x_scaler, "y_scaler": self.y_scaler,
                    "forecast_horizon": self.forecast_horizon, "feature_names_": self.feature_names_}, path)
        logger.info(f"Saved {self.model_name} to {path}")

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model = _LSTMNet(input_size=state["input_size"])
        self.model.load_state_dict(state["state_dict"])
        self.sequence_length = state["sequence_length"]
        self.x_scaler, self.y_scaler = state["x_scaler"], state["y_scaler"]
        self.forecast_horizon = state["forecast_horizon"]
        self.feature_names_ = state["feature_names_"]
        self.is_trained = True
        logger.info(f"Loaded {self.model_name} from {path}")