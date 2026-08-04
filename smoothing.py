"""
Temporal smoothing utilities.

The purpose of this module is to prevent unstable predictions
from changing every frame.
"""

from collections import deque, Counter


class PredictionSmoother:
    """
    Keeps recent predictions and returns the most common one.
    """

    def __init__(self, max_history=5):

        self.history = deque(
            maxlen=max_history
        )

    def update(self, value):

        if value is not None:
            self.history.append(value)

        return self.get_stable()

    def get_stable(self):

        if not self.history:
            return None

        counts = Counter(
            self.history
        )

        return counts.most_common(1)[0][0]

    def clear(self):

        self.history.clear()


class ConfidenceSmoother:
    """
    Smooths numerical confidence values.
    """

    def __init__(self, max_history=5):

        self.history = deque(
            maxlen=max_history
        )

    def update(self, value):

        if value is not None:
            self.history.append(float(value))

        if not self.history:
            return 0.0

        return sum(self.history) / len(
            self.history
        )

    def clear(self):

        self.history.clear()