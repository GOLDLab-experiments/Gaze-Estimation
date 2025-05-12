"""
utils.py contains utility functions and classes for the gaze estimation pipeline.
"""

class SmoothingFilter:
    def __init__(self, alpha=0.5):
        # Initialize the smoothing filter with a smoothing factor alpha (0 < alpha <= 1)
        # alpha controls how much weight is given to new values vs. previous smoothed value
        self.alpha = alpha
        self.prev = None

    def update(self, value):
        # Update the filter with a new value and return the smoothed result
        # If this is the first value, just store and return it
        if self.prev is None:
            self.prev = value
        else:
            # Apply exponential smoothing: new = alpha * value + (1-alpha) * prev
            self.prev = self.alpha * value + (1 - self.alpha) * self.prev
        return self.prev