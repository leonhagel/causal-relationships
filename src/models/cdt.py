import cdt
import time
import pandas as pd
import numpy as np
import networkx as nx
from castle.common import GraphDAG
import matplotlib.pyplot as plt

import utils

config = utils.json.load("../setup/config.json")
cdt.SETTINGS.rpath = config["Rscript"]

class Pipeline(utils.Pipeline):
    def __init__(self, model, init_kwargs):
        super().__init__(model, init_kwargs)
        self.params.pop("evaluate")
        self.G = None
        self.matrix = None
        self.features = None
        self.markov_blanket = None

    def fit(self, X, features, **kwargs):
        start = time.time()
        self.X_train = pd.DataFrame(X, columns=features)
        self.features = features
        self.params["fit"] = kwargs
        self.G = self.model.predict(self.X_train, **kwargs)
        self.matrix = nx.to_pandas_adjacency(self.G)
        self.time = time.time() - start

    def predict(self, y="TARGET", **kwargs):
        self.params["predict"] = {k:v for k,v in locals().items() if k not in ["self"]} 
        M = self.matrix 
        self.markov_blanket = utils.feature_selection.get_markov_blanket(M=M, y=y)
        return self.markov_blanket

    def plot(self, M=None, type="graph", features=None, threshold=None, edge_value=None, filepath=None, figsize=None, title=None, show=True, **kwargs):
        plt.figure(figsize=figsize);
        if M is None:
            M = self.matrix
        M = M.copy()
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



class PC(cdt.causality.graph.PC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)


class GES(cdt.causality.graph.GES):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)
        
class GIES(cdt.causality.graph.GIES):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)

class LiNGAM(cdt.causality.graph.LiNGAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)
