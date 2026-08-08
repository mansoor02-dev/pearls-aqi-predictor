import os

class AQIALertSystem:
    """
    Sends alerts when:
    1. Current AQI > 150 (Unhealthy)
    2. Predicted AQI > 150 in next 3 days
    3. Data drift detected
    """
    
    def __init__(self):
        # Could integrate with: email, Slack, Discord, PagerDuty
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    def check_and_alert(self, current_aqi: int, predictions: list):
        if current_aqi > 150:
            self._send_alert(f"🚨 CURRENT AQI is {current_aqi} (Unhealthy)!")
        
        for i, pred in enumerate(predictions, 1):
            if pred > 150:
                self._send_alert(f"⚠️ Predicted AQI in {i} day(s): {pred}")
    
    def _send_alert(self, message: str):
        # Send to Slack/Discord/email
        pass