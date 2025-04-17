"""
This module contains the EarlyStopping class which is used to stop training
when the loss does not improve for a certain number of epochs.
"""

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: int = 0, mode: str = 'min'):
        """
        Args:
            patience (int): How many epochs to wait for improvement.
            delta (int): Minimum change in monitored quantity to qualify as improvement.
            mode (str): 'min' for monitoring a decreasing value (e.g., loss),
                        'max' for monitoring an increasing value (e.g., accuracy).
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
        elif (self.mode == 'min' and value < self.best_value - self.delta) or \
             (self.mode == 'max' and value > self.best_value + self.delta):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
