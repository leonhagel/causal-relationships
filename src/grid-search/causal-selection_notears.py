import sys
sys.path.append(".")

import utils
import models
from utils import GridSearch


name_prefix = "causal-selection_notears"
predictive_models = {
    "boosting": "GradientBoostingClassifier",
    "decision-tree": "DecisionTreeClassifier",
    "logistic-regression": "LogisticRegression"
}
n_splits = 5

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
    # grid serach 
    for predictive_model in predictive_models.keys():
        name = f"{name_prefix}-{predictive_model}"
        print(f"[{name}] starting grid search...")
        # grid search parameter
        predictive_params = utils.json.load(f"../out/grid-search/{predictive_model}.json")["params"]["best"]
        train_test_split = {
            "to_woe": [to_woe],
            "scaler": "StandardScaler"
        }
        undersampling = {}
        init = {
            "causal_model": "Notears",
            "predictive_model": predictive_models[predictive_model],
            "causal_init": {
                "w_threshold": [0.3, 0.01, 0.1, 0.2, 0.5]
            },
            "predictive_init": predictive_params.get("init", {}),
        }
        fit = {
            "X_labels": [features],
            "causal_undersampling_ratio": [0.3, 0.01, 0.1, 0.2, None], 
            'predictive_undersampling_ratio': predictive_params.get("undersampling", {}).get("ratio", None),
            "predictive_params": predictive_params.get("fit", {})
        }
        ## 
        gs = GridSearch(models.CausalSelection, f"../out/grid-search/{name}.json")
        gs.fit(X, y, n_splits, train_test_split=train_test_split, undersampling=undersampling, init=init, fit=fit, verbose=2)