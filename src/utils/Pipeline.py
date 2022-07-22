from sklearn.metrics import roc_auc_score
import time
import numpy as np

class Pipeline:
    def __init__(self, model, init_kwargs):
        self.model = model
        self.params = {
            "init": init_kwargs,
            "fit": None, 
            "predict": None,
            "evaluate": None
        }
        self.X_train = None
        self.y_train = None
        self.score = roc_auc_score
        self.time = None

    def fit(self, X, y, **kwargs):
        start = time.time()
        self.X_train = X
        self.y_train = y
        self.params["fit"] = kwargs
        self.model.fit(X, y, **kwargs)
        self.time = time.time() - start

    def predict(self, X, **kwargs): 
        self.params["predict"] = kwargs
        return self.model.predict(X)

    def evaluate(self, y_true, y_hat, **kwargs):
        self.params["evaluate"] = kwargs
        return self.score(y_true, y_hat, **kwargs)

    def format_time(self, seconds=None):
        if isinstance(seconds, type(None)):
            seconds = self.time
        return f"{int(np.floor(seconds/60)):02d}:{int(seconds%60):02d}"