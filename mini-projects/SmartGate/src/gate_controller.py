class GateController:
    def __init__(self, min_safe_height=120):
        self.min_safe_height = min_safe_height
        self.status = "CLOSED"

    def check_and_open(self, height):
        if height >= self.min_safe_height:
            self.status = "OPEN"
        else:
            self.status = "CLOSED"
        return self.status
