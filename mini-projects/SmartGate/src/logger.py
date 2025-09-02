import datetime

class Logger:
    def __init__(self):
        self.logs = []

    def log_attempt(self, height, status):
        entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "height": height,
            "status": status
        }
        self.logs.append(entry)

    def show_logs(self):
        for log in self.logs:
            print(log)
