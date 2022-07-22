import castle.algorithms
from castle.common import GraphDAG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
import time

import utils

class Pipeline(utils.Pipeline):
    def __init__(self, model, init_kwargs):
        super().__init__(model, init_kwargs)
        self.params.pop("evaluate")
        self.matrix = None
        self.features = None
        self.thresholds = []
        self.markov_blanket = None
    
    def fit(self, X, features, threshold=None, **kwargs):
        start = time.time()
        self.X_train = X
        self.features = features
        self.params["fit"] = kwargs
        self.model.learn(X, **kwargs)
        self.matrix = pd.DataFrame(self.model.causal_matrix, index=self.features, columns=self.features)
        self.thresholds = utils.h_value.get_thresholds(self.matrix.values)
        if threshold is None:
            try:
                threshold = self.model.w_threshold
            except AttributeError:
                threshold = 0
        elif any([threshold<0, threshold>1]):
            threshold = self.thresholds[threshold]
        self.threshold = threshold
        self.time = time.time() - start
    
    def predict(self, y="TARGET", threshold=None):
        self.params["predict"] = {k:v for k,v in locals().items() if k not in ["self"]}
        if threshold is None:
            threshold = self.threshold
        self.markov_blanket = utils.feature_selection.get_markov_blanket(M=self.matrix, y=y, threshold=threshold)
        return self.markov_blanket

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
            plt.title(title)
        if filepath is not None:
            plt.savefig(filepath)
            print(f"stored plot to {filepath}.")
        if show:
            plt.show()
    
    def generate_plots(self, filepath, matrix=None, markov_blanket=None, colors=None, target="TARGET", edge_value=.001, show=True, **kwargs):
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
        if markov_blanket is None:
            markov_blanket = self.markov_blanket
        causal_features = utils.feature_selection.get_causal_features(markov_blanket=markov_blanket)
        features = [feature for feature in M.columns if feature not in [target]+causal_features] + causal_features + [target]
        colors = [colors[target] if feature == target else colors["causal_feature"] if feature in causal_features else colors["other"] for feature in features]
        self.plot(M=M.loc[features, features], edge_value=edge_value, features=features, node_color=colors, filepath=filepath, show=show, **kwargs)


class GOLEM(castle.algorithms.GOLEM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class ANMNonlinear(castle.algorithms.ANMNonlinear):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class CORL(castle.algorithms.CORL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class DirectLiNGAM(castle.algorithms.DirectLiNGAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class GraNDAG(castle.algorithms.GraNDAG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class ICALiNGAM(castle.algorithms.ICALiNGAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class MCSL(castle.algorithms.MCSL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class Notears(castle.algorithms.Notears):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class NotearsNonlinear(castle.algorithms.NotearsNonlinear):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class PC(castle.algorithms.PC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class RL(castle.algorithms.RL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class TTPM(castle.algorithms.TTPM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)
