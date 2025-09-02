class AlertSystem:
    def __init__(self):
        self.alerts = []

    def trigger_alert(self, height):
        alert_msg = f"⚠️ ALERT: Child of height {height} cm tried to exit!"
        self.alerts.append(alert_msg)
        print(alert_msg)
