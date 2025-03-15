class EarlyStopping:
    def __init__(self, patience: int = 5, delta: int = 0):
        self.patience = patience # How many epochs to wait for improvement
        self.delta = delta # Minimum change in monitored quantity to qualify as improvement
        self.counter = 0 # Counter for patience
        self.best_loss= None # Best score so far
        self.early_stop = False # Whether to stop training

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss # Set best less to the first loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training
