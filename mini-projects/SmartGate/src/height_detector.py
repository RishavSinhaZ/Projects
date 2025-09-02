import random

class HeightDetector:
    def __init__(self, min_height=50, max_height=200):
        self.min_height = min_height
        self.max_height = max_height

    def detect_height(self):
        # Simulate by generating a random height (in cm)
        return random.randint(self.min_height, self.max_height)
