import sys
sys.path.append(".")

import utils
import models
from utils import GridSearch

name = "boosting"
selection_models = ["correlation", "ols", "sfs", "rfe"]

# loading the data
to_dummy = [
    "prop_type",
    "flag_fthb",
    "pgrm_ind",
    "occpy_sts",
    "loan_purpose",
    "channel",
    "prod_type"
]
to_woe = [
    "zipcode",
    "cd_msa",
    "st",
    "seller_name",
    "servicer_name",
    "prop_val_meth",
]
to_woe += to_dummy
to_dummy = []
drop = ["id_loan"]
target = "TARGET"

if __name__ == "__main__":
    # loading the data
    train = utils.csv.read("../data/features_train_pre-tts_no-dummies.csv", low_memory=False).dropna()
    train = train.drop(drop, axis=1)
    y = train[target]
    X = train.drop(target, axis=1)
    features = list(X.columns)
    # grid search parameter
    train_test_split = {
        "to_woe": [to_woe],
        "scaler": "StandardScaler"
    }
    undersampling = {}
    init = {
        "learning_rate": [0.1, 0.01, 0.001],
        "subsample": [1, 0.25, 0.5],
        "min_samples_leaf": [1],
        "max_depth": [3, 2],
        "max_features": [None, "sqrt"]
    }
    fit = {}
    # grid search
    gs = GridSearch(models.sklearn.GradientBoostingClassifier, f"../out/grid-search/{name}.json")
    gs.fit(X, y, 5, train_test_split=train_test_split, undersampling=undersampling, init=init, fit=fit, verbose=2)
    for selection_model in selection_models:
        # exclude rfe and sfs from grid search: due to extreme training time
        if selection_model in ["sfs", "rfe"]:
            selection_init = {"selection_model": selection_model, "predictive_model": "GradientBoostingClassifier", "selection_init": {"cv": 2}, "predictive_init":{}}
            selection_fit = {"predictive_params": {}}
        else:
            selection_init = {"selection_model": selection_model, "predictive_model": "GradientBoostingClassifier", "predictive_init": init}
            selection_fit = {"predictive_params": fit}
        gs = GridSearch(models.FeatureSelection, f"../out/grid-search/feature-selection_{selection_model}-{name}.json")
        gs.fit(X, y, 5, train_test_split=train_test_split, undersampling=undersampling, init=selection_init, fit=selection_fit, verbose=2)