import utils
import models

import time
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np


selection_models = {
    "sfs": SFS,
    "rfe": RFECV, 
    "ols": sm.OLS,
    "correlation": "correlation",
}

predictive_models = {
    "DecisionTreeClassifier": models.sklearn.DecisionTreeClassifier,
    "GradientBoostingClassifier": models.sklearn.GradientBoostingClassifier,
    "LogisticRegression": models.sklearn.LogisticRegression, 
}


class FeatureSelection:
    def __init__(self, selection_model, predictive_model, selection_init={}, predictive_init={}):
        self.selection_model = selection_models[selection_model]
        self.predictive_model = predictive_models[predictive_model](**predictive_init)
        self.pipeline = Pipeline(self, {k: v for k, v in locals().items() if k != "self"})
        
class Pipeline(utils.Pipeline):
    def __init__(self, model, init_kwargs):
        super().__init__(model, init_kwargs)
        self.model = model
        self.selection_name = init_kwargs.get("selection_model", None)
        self.prediction_name = init_kwargs.get("prediction_model", None)
        self.selected_features = None
        self.deleted_features = None
        self.X_train = {}
        self.y_train = {}
        self.evaluate = self.model.predictive_model.pipeline.evaluate
    
    def fit(self, X_train, y_train, selection_params={}, predictive_params={}):
        start = time.time()
        self.params["fit"] = {
            "selection_params": selection_params, 
            "predictive_params": selection_params
        }
        features, _ = self.get_features(X_train, y_train, **selection_params)
        self.X_train["predictive"] = X_train.loc[:, features]
        self.y_train["predictive"] = y_train
        self.model.predictive_model.pipeline.fit(X_train.loc[:, features], y_train, **predictive_params)
        self.time = time.time() - start
    
    def predict(self, X, **kwargs):
        self.params["predict"] = kwargs
        features = self.selected_features
        return self.model.predictive_model.pipeline.predict(X.loc[:, features], **kwargs)
              
    def get_features(self, X_train, y_train, **kwargs):
        fn = {
            "sfs": self.sfsFeature,
            "rfe": self.rfeFeature, 
            "ols": self.olsFeature,
            "correlation": self.correlationFeature,
        }
        self.selected_features, self.deleted_features = fn[self.selection_name](X_train, y_train, **kwargs)
        return self.selected_features, self.deleted_features
        
    # correlation without dealing with multicollinearity
    # df is the train dataset including target
    def correlationFeature(self, X_train, y_train, show=False):
        self.X_train["selection"] = X_train
        self.y_train["selection"] = y_train
        X = X_train.copy()
        X["TARGET"] = y_train
        corr = X.corr()
        if show:
            plt.figure(figsize=(16, 6))
            heatmap = sns.heatmap(corr,  xticklabels = corr.columns, yticklabels = corr.columns)
            plt.show()
        # get the mean and standard deviation of the correlation to set the threshold
        mean_corr_target = np.mean(abs(corr['TARGET']))
        sd_corr_target = np.std(corr['TARGET'])
        # delete features that are one std away from the mean
        corr_features_delete = list(corr[(corr['TARGET'] > (mean_corr_target + sd_corr_target))
               & (corr['TARGET'] < (mean_corr_target - sd_corr_target))].index)
        corr_features_select = [feature for feature in corr.columns if feature not in ["TARGET"]+corr_features_delete]
        return corr_features_select, corr_features_delete

    # Recursive Feature Elimination (RFE)
    def rfeFeature(self, X_train, y_train, show=False, **kwargs):
        init = {"step":1, "cv": 2, "scoring":'accuracy'}
        init.update(self.params["init"]["selection_init"])
        init.update(kwargs)
        self.params["init"]["selection_init"] = init
        self.X_train["selection"] = X_train
        self.y_train["selection"] = y_train
        rfecv = self.model.selection_model(estimator = self.model.predictive_model, **init)
        rfecv_result = rfecv.fit(X_train, np.ravel(y_train.values))
        num_feature_selection = rfecv.n_features_
        rfe_feature_select = list(X_train.columns[rfecv.support_])
        rfe_feature_delete = list(set(X_train.columns.tolist()) - set(rfe_feature_select))
        if show:
            plt.figure(figsize=(16, 6))
            plt.title('Recursive Feature Elimination with Cross-Validation')
            plt.xlabel('Number of Features')
            plt.ylabel('Performance (% Correctly Classified)')
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth=3)
            plt.legend(['1st cv', '2nd cv'], loc='upper left')
            plt.show()
        return rfe_feature_select, rfe_feature_delete

    # Sequential Feature Selection (SFS)
    def sfsFeature(self, X_train, y_train, show=False, **kwargs):
        init = {"forward": True, "floating": False, "scoring": 'accuracy', "cv": 3}
        init.update(self.params["init"]["selection_init"])
        init.update(kwargs)
        self.params["init"]["selection_init"] = init
        self.X_train["selection"] = X_train
        self.y_train["selection"] = y_train
        sfs = self.model.selection_model(estimator = self.model.predictive_model, k_features=(2, X_train.shape[1]), **init)
        sfs = sfs.fit(X_train, np.ravel(y_train.values))
        sfs_feature_select = list(sfs.k_feature_names_)
        sfs_feature_delete = list(set(X_train.columns.tolist()) - set(sfs_feature_select))
        if show:
            plt.figure(figsize=(20, 12))
            plot_sfs(sfs.get_metric_dict())
            plt.title('Sequential Forward Selection')
            plt.show()
        return sfs_feature_select, sfs_feature_delete

    # regression model (choose variables that are significant)
    def olsFeature(self, X_train, y_train, level=0.001):
        self.X_train["selection"] = X_train
        self.y_train["selection"] = y_train
        ols_model = self.model.selection_model(y_train, X_train, **self.params["init"]["selection_init"]).fit()
        ols_summary = ols_model.summary()
        ols_pval = pd.DataFrame(ols_model.pvalues)
        # delete insignificant feature
        ols_model_features_delete = list(ols_pval[abs(ols_pval[0])<=level].index)
        ols_model_features_select = [feature for feature in X_train.columns if feature not in ols_model_features_delete]
        return ols_model_features_select, ols_model_features_delete