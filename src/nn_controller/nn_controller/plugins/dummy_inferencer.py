import numpy as np

class DummyInferencer:
    def __init__(self, logger):
        self.logger = logger
        self.k = 0.002

    def infer(self, bgr_image):
        mean_val = float(np.mean(bgr_image))
        steer = self.k * (mean_val - 127.0)
        return max(-1.0, min(1.0, steer))
