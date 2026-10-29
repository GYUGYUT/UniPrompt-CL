class EarlyStopping:
    def __init__(self, patience=5, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.mode == "min":
            if self.best_score is None or current_score < self.best_score:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == "max":
            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True