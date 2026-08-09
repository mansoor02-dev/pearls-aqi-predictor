import json
import smtplib
import textwrap
from datetime import datetime
from email.mime.text import MIMEText
from typing import List, Optional

import requests

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# AQI level → human label
_AQI_LABELS = [
    (50,  "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (500, "Hazardous"),
]


def _aqi_label(aqi: float) -> str:
    for ceiling, label in _AQI_LABELS:
        if aqi <= ceiling:
            return label
    return "Hazardous"


class AQIAlertSystem:
    """
    Sends alerts when:
      1. Current AQI exceeds AQI_ALERT_THRESHOLD (default 150 — Unhealthy).
      2. Any forecast day's AQI exceeds the threshold.
      3. Data drift is detected above DRIFT_THRESHOLD.

    All alerts are logged regardless of channel config.
    Slack / Email only fire when the relevant env vars are present.
    """

    def __init__(self):
        self.threshold = settings.AQI_ALERT_THRESHOLD
        self.slack_url = settings.SLACK_WEBHOOK_URL
        self._smtp_ready = bool(
            settings.SMTP_HOST and settings.ALERT_EMAIL_TO and settings.ALERT_EMAIL_FROM
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def check_and_alert(
        self,
        current_aqi: float,
        predictions: List[float],
        city: str | None = None,
    ) -> None:
        """
        Check current + forecast AQI values and fire alerts if needed.

        Parameters
        ----------
        current_aqi  : Latest observed AQI reading.
        predictions  : List of forecast values (one per day ahead, e.g. [d1, d2, d3]).
        city         : Optional city name for the alert message.
        """
        city = city or settings.CITY

        if current_aqi > self.threshold:
            label = _aqi_label(current_aqi)
            self._send_alert(
                title=f"🚨 Current AQI Alert — {city}",
                message=(
                    f"Current AQI in {city} is *{current_aqi:.0f}* ({label}), "
                    f"which exceeds the alert threshold of {self.threshold}.\n"
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                ),
            )

        for i, pred in enumerate(predictions, start=1):
            if pred is not None and pred > self.threshold:
                label = _aqi_label(pred)
                self._send_alert(
                    title=f"⚠️ Forecast AQI Alert — {city} (Day {i})",
                    message=(
                        f"Forecast AQI for {city} in {i} day(s) is *{pred:.0f}* ({label}), "
                        f"exceeding the threshold of {self.threshold}.\n"
                        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ),
                )

    def send_drift_alert(
        self,
        drifted_features: List[str],
        share_drifted: float,
        city: str | None = None,
    ) -> None:
        """
        Fire an alert when data drift is detected.

        Parameters
        ----------
        drifted_features : List of feature column names that drifted.
        share_drifted    : Fraction of all features that drifted (0–1).
        city             : Optional city name for the alert message.
        """
        city = city or settings.CITY
        pct = f"{share_drifted * 100:.1f}%"
        feat_str = ", ".join(drifted_features) if drifted_features else "none identified"

        self._send_alert(
            title=f"📊 Data Drift Detected — {city}",
            message=(
                f"*{pct}* of input features have drifted in {city}.\n"
                f"Drifted columns: {feat_str}\n"
                f"Consider re-training the model.\n"
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ),
        )

    def send_model_alert(self, horizon: int, rmse: float, baseline_rmse: float) -> None:
        """
        Fire an alert when a live model's RMSE degrades beyond 20% of its
        baseline (training) RMSE — a sign the model needs retraining.
        """
        degradation = (rmse - baseline_rmse) / baseline_rmse * 100
        if degradation > 20:
            self._send_alert(
                title=f"🤖 Model Performance Degradation — Horizon {horizon}d",
                message=(
                    f"Live RMSE for the {horizon}-day model is *{rmse:.2f}* "
                    f"(+{degradation:.1f}% above training baseline of {baseline_rmse:.2f}).\n"
                    f"Consider triggering the training pipeline.\n"
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                ),
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _send_alert(self, title: str, message: str) -> None:
        """
        Fan-out to all configured channels. Logger fires unconditionally.
        Slack / Email only fire when their respective settings are present.
        Failures in one channel never block the others.
        """
        full = f"{title}\n{message}"
        logger.warning(full)
        self._try_slack(title, message)
        self._try_email(title, message)

    def _try_slack(self, title: str, message: str) -> None:
        if not self.slack_url:
            return
        payload = {
            "attachments": [
                {
                    "color": "#ff4d4d",
                    "title": title,
                    "text": message,
                    "footer": "Pearls AQI Predictor",
                    "ts": int(datetime.now().timestamp()),
                }
            ]
        }
        try:
            resp = requests.post(
                self.slack_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            resp.raise_for_status()
            logger.info("Slack alert sent successfully.")
        except Exception as exc:
            logger.error(f"Slack alert failed: {exc}")

    def _try_email(self, title: str, message: str) -> None:
        if not self._smtp_ready:
            return
        plain = textwrap.dedent(message).replace("*", "")   # strip Slack markdown
        msg = MIMEText(plain, "plain", "utf-8")
        msg["Subject"] = f"[AQI Alert] {title}"
        msg["From"] = settings.ALERT_EMAIL_FROM
        msg["To"] = settings.ALERT_EMAIL_TO
        try:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
                server.ehlo()
                server.starttls()
                if settings.SMTP_USER and settings.SMTP_PASSWORD:
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.ALERT_EMAIL_FROM, [settings.ALERT_EMAIL_TO], msg.as_string())
            logger.info(f"Email alert sent to {settings.ALERT_EMAIL_TO}.")
        except Exception as exc:
            logger.error(f"Email alert failed: {exc}")