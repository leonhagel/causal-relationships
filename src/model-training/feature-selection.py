import sys
sys.path.append(".")

import models
import utils
import matplotlib.pyplot as plt

selection_models = ["correlation", "ols", "rfe", "sfs"]
predictive_models = ["decision-tree", "logistic-regression", "boosting"]
translation = {
    "boosting": "GradientBoostingClassifier", 
    "decision-tree": "DecisionTreeClassifier", 
    "logistic-regression": "LogisticRegression"
}
name_prefix = "feature-selection"
p_name = "best"

drop = ["id_loan"]
target = "TARGET"

if __name__ == "__main__":
    # loading the data
    train = utils.csv.read("../data/features_train_no-dummies.csv").dropna()
    test = utils.csv.read("../data/features_test_no-dummies.csv")
    X_train = train.drop(drop + [target], axis=1)
    X_test = test.drop(drop, axis=1)
    y_train = train[target]
    for predictive_model in predictive_models:
        for selection_model in selection_models:
            name = f"{name_prefix}_{selection_model}-{predictive_model}"
            params = utils.json.load(f"../out/grid-search/{name}.json")["params"][p_name]
            # model training
            print(f"[{name}] starting model training... (data shapes: {[X_train.shape, y_train.shape, X_test.shape]})")
            model = models.FeatureSelection(**params.get("init", {}))
            model.pipeline.fit(X_train, y_train, **params.get("fit", {}))
            train_score = model.pipeline.evaluate(y_train, model.pipeline.predict(X_train), **params.get("predict", {}))
            print(f"[{name}] train score:    {train_score:.4f}")
            out = test.copy()
            out[target] = model.pipeline.predict(X_test, **params.get("predict", {}))
            utils.csv.dump(out.loc[:, ["id_loan", target]], f"../out/predictions/{name}.csv")
            stats = {
                "train_score": train_score,
                "time": {
                    "total": [model.pipeline.format_time(), model.pipeline.time],
                    "predictive": [model.predictive_model.pipeline.format_time(), model.predictive_model.pipeline.time],
                },
                "shapes": {
                    "selection": [model.pipeline.X_train["selection"].shape, model.pipeline.y_train["selection"].shape],
                    "predictive": [model.pipeline.X_train["predictive"].shape, model.pipeline.y_train["predictive"].shape]
                },
                "params": model.pipeline.params
            } 
            utils.json.dump(stats, f"../out/training-stats/{name}.json", verbose=1)
            utils.json.dump(model.pipeline.selected_features, f"../out/selected-features/{name}.json", verbose=1)