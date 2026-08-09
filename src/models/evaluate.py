import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from src.utils.logger import setup_logger

logger = setup_logger("Evaluation")

def score_predictions(name, horizon, y_test_raw, pred_raw, y_naive):
    r2 = r2_score(y_test_raw, pred_raw)
    mae = mean_absolute_error(y_test_raw, pred_raw)
    mse = mean_squared_error(y_test_raw, pred_raw)
    rmse = root_mean_squared_error(y_test_raw, pred_raw)

    skill = None
    if y_naive is not None:
        naive_rmse = root_mean_squared_error(y_test_raw, y_naive)
        skill = 1 - (rmse / naive_rmse) if naive_rmse > 0 else np.nan

    metrics = {
        "model": name, "horizon": f"{horizon}d",
        "r2": r2, "mae": mae, "mse": mse, "rmse": rmse, "skill_vs_naive": skill,
    }
    skill_str = f"   Skill: {skill:+.3f}" if skill is not None else ""
    logger.info(f"{name:32s}  R2: {r2:.4f}   MAE: {mae:.3f}   RMSE: {rmse:.3f}{skill_str}")
    return metrics