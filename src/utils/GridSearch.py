import utils 
from sklearn.model_selection import StratifiedKFold
import itertools
import numpy as np
import matplotlib.pyplot as plt

class GridSearch:
    def __init__(self, model_class, filepath=None):
        self.X = None
        self.y = None
        self.model_class = model_class
        self.n_splits = None
        self.params = {}
        self.train_scores = {}
        self.test_scores = {}
        self.scores = {}
        self.filepath = filepath
        self.errors = {}
        if filepath is not None:
            self.load(filepath)

    def fit_StratifiedKFold(self, X, y, n_splits, model_name="kfold", train_test_split={}, undersampling={}, init={}, fit={}, predict={}, verbose=1, **kwargs):
        self.n_splits = n_splits
        self.params[model_name] = {
            "train_test_split": train_test_split, 
            "undersampling": undersampling,
            "init": init, 
            "fit": fit, 
            "predict": predict, 
        }
        train_score = []
        test_score = []
        counter = 1 
        if n_splits==1:
            indicies = [[None, None]]
        else:
            skf = StratifiedKFold(n_splits=n_splits, **kwargs)
            indicies = skf.split(X, y)
        for train_index, test_index in indicies:
            if verbose >= 1:
                print(f"starting fold {counter}...")  
            X_train, X_test, y_train, y_test = utils.train_test_split(X, y, train_index=train_index, test_index=test_index, **train_test_split)
            if undersampling != {}:
                X_train, y_train = utils.undersampling.basic(X_train, y_train, **undersampling)
                X_test, y_test = X_test.values, y_test.values.reshape(-1, 1)
            if verbose >= 2:    
                print(f"[fold {counter}] data shapes: {[X_train.shape, X_test.shape, y_train.shape, y_test.shape]}")
            model = self.model_class(**init)
            model.pipeline.fit(X_train, y_train, **fit)
            train_score += [model.pipeline.evaluate(y_train, model.pipeline.predict(X_train, **predict))]
            test_score += [model.pipeline.evaluate(y_test, model.pipeline.predict(X_test, **predict))]
            if verbose >= 1:
                print(f"[fold {counter}] train score: {train_score[-1]:.4f}    test score: {test_score[-1]:.4f}")
            counter += 1
        self.train_scores[model_name] = np.array(train_score)
        self.test_scores[model_name] = np.array(test_score)
    
    def get_parameter_combinations(self, parameter_values):
        params = {key: values if not isinstance(values, dict) else self.get_parameter_combinations(values) for key, values in parameter_values.items()}
        params = {k: [v] if not isinstance(v, list) else v for k, v in params.items()}
        params = {key: [{key: v} for v in values] for key, values in params.items()}
        parameter_combinations = list(itertools.product(*list(params.values())))
        if parameter_combinations == [()]:
            parameter_combinations = [{}]
        parameter_combinations = [{k: v for d in params for k, v in d.items()} for params in parameter_combinations]
        return parameter_combinations

    def fit(self, X, y, n_splits, model_name_prefix="gs", train_test_split={}, undersampling={}, init={}, fit={}, predict={}, dump=True, verbose=1, **kwargs):
        params = {
            "train_test_split": train_test_split,
            "undersampling": undersampling,
            "init": init,
            "fit": fit,
            "predict": predict,
        }
        param_combinations = self.get_parameter_combinations(params)
        if verbose >= 1:
            print(f"[grid search] {len(param_combinations)} parameter combinations identified.")
        counter = 1
        model_counter = 1 + max([0] + [int(key.replace(model_name_prefix+"_", "")) for key in self.params.keys() if model_name_prefix in key])
        params2name = {str(v): k for k, v in self.params.items()}
        for params in param_combinations:
            if params not in self.params.values():
                if verbose >= 1:
                    print(f"[grid search] starting stratified kfold for parameter combination {counter}...")
                model_name = "_".join([model_name_prefix, str(model_counter)])
                try:
                    self.fit_StratifiedKFold(X=X, y=y, n_splits=n_splits, model_name=model_name, verbose=verbose-1, **params)  
                except Exception as e:
                    self.errors[model_name] = [e, params]
                    print(f"[grid search] error occured during stratified kfold. ({e})")
                if dump:
                    self.dump(verbose=verbose-1)
                model_counter += 1
            else:
                print(f"[grid search] skipped parameter combination, already used in '{params2name[str(params)]}'.")
            counter += 1
    def set_best(self, best):
        self.params["best"] = self.params[best]
        self.test_scores["best"] = self.test_scores[best]
        self.train_scores["best"] = self.train_scores[best]
    
    def evaluate(self):
        scores = {}
        for model in self.test_scores.keys():
            scores[model] = (self.train_scores[model].mean() - self.test_scores[model].mean()) * self.test_scores[model].std() * (1 - self.test_scores[model]).mean()
        best_model = list(scores.keys())[np.argmin(list(scores.values()))]
        self.scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
        self.set_best(best_model)
        return self.scores
    
    def plot(self, sort_values=True, figsize=None):
        plt.figure(figsize=figsize);
        if sort_values:
            scores = {scores.mean(): name for name, scores in self.test_scores.items()}
            sorted_scores = {scores[score]: score for score in sorted(scores, reverse=False)}
            data = [self.test_scores[name] for name in sorted_scores.keys()]
            keys = sorted_scores.keys()
        else:
            data = list(self.test_scores.values())
            keys = self.test_scores.keys()
        labels = [f"{model}\ntest score" for model in keys]
        plt.boxplot(data,labels=labels, positions=range(len(labels)));
        plt.scatter(labels, [self.train_scores[key].mean() for key in keys], marker="s", label="train scores", c="tab:cyan");
        plt.legend();
        plt.show();


    def load(self, filepath):
        try:
            dump = utils.json.load(filepath)
            self.params = dump["params"]
            self.train_scores = {k: np.array(v) for k, v in dump["train_scores"].items()}
            self.test_scores = {k: np.array(v) for k, v in dump["test_scores"].items()}
            print(f"finished loading results from {filepath}.")
        except FileNotFoundError:
            pass

        
    def dump(self, filepath=None, verbose=1):
        if filepath is None:
            filepath = self.filepath
        dump = {
            "params": {k: self.params[k] for k in sorted(self.params)},
            "train_scores": {k: self.train_scores[k].tolist() for k in sorted(self.train_scores)},
            "test_scores": {k: self.test_scores[k].tolist() for k in sorted(self.test_scores)},
        }
        utils.json.dump(dump, filepath, verbose=verbose)