import sys
sys.path.append(".")

import utils
import models
from utils import GridSearch


name = "castle"

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
        "reg_lambda": [1, 3, 8, 15],
        "reg_beta": [1, 3, 8, 15],
        "w_threshold": -3, 
        "sigmoid_output": [[0]],
        "ckpt_file": "../cache/tmp.ckpt",
        "loss_function": "bce"
    }
    fit = {
        "X_labels": [features],
        "undersampling_ratio": [1/3, 1/10, 1/25, None],
        "verbose": 0
    }
    gs = GridSearch(models.CASTLE, f"../out/grid-search/{name}.json")
    gs.fit(X, y, 5, train_test_split=train_test_split, undersampling=undersampling, init=init, fit=fit, verbose=2)