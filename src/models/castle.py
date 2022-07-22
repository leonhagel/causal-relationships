import sys
sys.path = [path for path in sys.path if "grid-search" not in path]

import utils
from CASTLE import CASTLE as CASTLE_base
import time
import numpy as np
import pandas as pd
import networkx as nx
from castle.common import GraphDAG
import matplotlib.pyplot as plt

class Pipeline(utils.Pipeline):
    def __init__(self, model, init_kwargs):
        super().__init__(model, init_kwargs)
        self.X_val = None
        self.y_val = None
        self.matrix = None
        self.thresholds = None
        self.threshold = None
        self.features = None
        self.markov_blanket = None
    
    def fit(self, X, y, X_labels=None, y_label="TARGET", val_size=0.2, undersampling_ratio=None, threshold=None, build={}, **kwargs):
        start = time.time()
        self.params["fit"] = {k:v for k,v in locals().items() if k not in ["self", "X", "y", "start"]}
        self.features = [y_label] + X_labels
        X_train, X_val, y_train, y_val = utils.train_test_split(X, y, val_size)
        #if undersampling_ratio is not None:
        X_train, y_train = utils.undersampling.basic(X_train, y_train, undersampling_ratio)
        if len(y_train.shape) != 2:
            y_train = y_train.reshape(-1, 1)
        if len(y_val.shape) != 2:
            try:
                y_val = y_val.reshape(-1, 1)
            except AttributeError:
                X_val = X_val.values
                y_val = y_val.values.reshape(-1, 1)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model.build(np.hstack([y_train, X_train]), **build)
        self.model.fit(np.hstack([y_train, X_train]), y_train, np.hstack([y_val, X_val]), y_val, **kwargs)
        self.matrix = pd.DataFrame(self.model.W.eval(session=self.model.sess), index=self.features, columns=self.features)
        self.thresholds = utils.h_value.get_thresholds(self.matrix.values)
        if threshold is None:
            threshold = self.model.w_threshold 
        if any([threshold<0, threshold>1]):
            threshold = self.thresholds[threshold]
        self.threshold = threshold
        self.markov_blanket = utils.feature_selection.get_markov_blanket(self.matrix, threshold=threshold)
        self.time = time.time() - start
    
    def predict(self, X, **kwargs):
        self.params["predict"] = kwargs
        return self.model.pred(X)

    def plot(self, M=None, type="graph", features=None, threshold=None, edge_value=None, filepath=None, figsize=None, title=None, show=True, **kwargs):
        plt.figure(figsize=figsize);
        if M is None:
            M = self.matrix
        M = M.copy()
        if threshold is not None:
            if any([threshold > 1, threshold < 0]):
                threshold = self.thresholds[threshold]
            M.loc[:, :] = np.where(M.values > threshold, M.values, 0)
        if edge_value is not None:
            M.loc[:, :] = np.where(M.values > 0, edge_value, 0)
        if features is None:
            features = M.columns
        if type == "graph":
            G = nx.DiGraph(M.loc[features, features])
            nx.draw_networkx(G, **kwargs)
        elif type == "matrix":
            GraphDAG(M.values, **kwargs)
        else:
            raise ValueError("'type' not recognized, supported types: ['graph', 'matrix']")
        if title is not None:
            print(title)
            plt.title(title)
        if filepath is not None:
            plt.savefig(filepath)
            print(f"stored plot to {filepath}.")
        if show:
            plt.show()

    def generate_plots(self, filepath, matrix=None, markov_blanket=None, colors=None, target="TARGET", edge_value=.001, threshold=None, show=True, **kwargs):
        if colors is None:
            colors = {
                target: "tab:olive", 
                "causal_feature": "tab:cyan", 
                "other": "lightgray"
            }
        plt.clf()
        if matrix is None:
            matrix = self.matrix
        M = matrix.copy()
        if threshold is not None:
            if any([threshold > 1, threshold < 0]):
                threshold = self.thresholds[threshold]
        else:
            threshold = self.threshold
        if markov_blanket is None:
            markov_blanket = utils.feature_selection.get_markov_blanket(self.matrix, threshold=threshold)
        causal_features = utils.feature_selection.get_causal_features(markov_blanket=markov_blanket)
        features = [feature for feature in M.columns if feature not in [target]+causal_features] + causal_features + [target]
        colors = [colors[target] if feature == target else colors["causal_feature"] if feature in causal_features else colors["other"] for feature in features]
        self.plot(M=M.loc[features, features], edge_value=edge_value, features=features, node_color=colors, filepath=filepath, threshold=threshold, show=show, **kwargs)


class CASTLE(CASTLE_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)
