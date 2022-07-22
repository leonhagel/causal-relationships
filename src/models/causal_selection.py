import models.sklearn
import models.gcastle
import models.cdt
import utils
import time

import numpy as np

causal_models = {
    "PC": models.cdt.PC,
    "GES": models.cdt.GES,
    "GIES": models.cdt.GIES,
    "LiNGAM": models.cdt.LiNGAM,
    "ICALiNGAM": models.gcastle.ICALiNGAM,
    "Notears": models.gcastle.Notears, 
    "GOLEM": models.gcastle.GOLEM
}

predictive_models = {
    "GradientBoostingClassifier": models.sklearn.GradientBoostingClassifier,
    "RandomForestClassifier": models.sklearn.RandomForestClassifier,
    "DecisionTreeClassifier": models.sklearn.DecisionTreeClassifier,
    "LogisticRegression": models.sklearn.LogisticRegression
}

class Pipeline(utils.Pipeline):
    def __init__(self, model, init_kwargs):
        super().__init__(model, init_kwargs)
        self.y_label = None
        self.plot = self.model.causal_model.pipeline.plot 
        self.generate_plots = self.model.causal_model.pipeline.generate_plots
        self.matrix = None
        self.markov_blanket = None

    def fit(self, X, y, X_labels, y_label="TARGET", threshold=0, causal_undersampling_ratio=None, causal_params={}, predictive_undersampling_ratio=None, predictive_params={}, random_state=123):
        start = time.time()
        self.X_train = {}
        self.y_train = {}
        self.y_label = y_label
        self.params["fit"] = {k:v for k,v in locals().items() if k not in ["self", "X", "y", "start"]}
        # fit causal model 
        if not isinstance(causal_undersampling_ratio, type(None)):
            X_causal, y_causal = utils.undersampling.basic(X, y, causal_undersampling_ratio, random_state)
        else:
            try:
                X_causal, y_causal = X.values, y.values.reshape(-1, 1)
            except AttributeError:
                X_causal, y_causal = X, y
        self.X_train["causal"] = X_causal
        self.y_train["causal"] = y_causal
        self.model.causal_model.pipeline.fit(np.hstack([y_causal, X_causal]), features=[y_label]+X_labels, **causal_params)
        self.markov_blanket = self.model.causal_model.pipeline.predict(y=y_label, threshold=threshold) 
        self.matrix = self.model.causal_model.pipeline.matrix
        # fit predictive model
        causal_features = utils.feature_selection.get_causal_features(markov_blanket=self.markov_blanket, y=y_label, threshold=threshold)
        if not isinstance(predictive_undersampling_ratio, type(None)):
            X_predictive, y_predictive = utils.undersampling.basic(X.loc[:, causal_features], y, predictive_undersampling_ratio, random_state)
        else:
            X_predictive, y_predictive = X.loc[:, causal_features], y
        self.X_train["predictive"] = X_predictive
        self.y_train["predictive"] = y_predictive
        self.model.predictive_model.pipeline.fit(X_predictive, y_predictive, **predictive_params)
        self.time = time.time() - start
        
    def predict(self, X, **kwargs):
        self.params["predict"] = kwargs
        causal_features = utils.feature_selection.get_causal_features(markov_blanket=self.markov_blanket, y=self.y_label)
        return self.model.predictive_model.pipeline.predict(X.loc[:, causal_features], **kwargs)



class CausalSelection:
    def __init__(self, causal_model, predictive_model, causal_init={}, predictive_init={}):
        self.causal_model = causal_models[causal_model](**causal_init)
        self.predictive_model = predictive_models[predictive_model](**predictive_init)
        self.pipeline = Pipeline(self, {k: v for k, v in locals().items() if k != "self"})
