import utils
import sklearn.ensemble
import sklearn.tree
import sklearn.linear_model


class Pipeline(utils.Pipeline):
    def __init__(self, model, init_kwargs):
        super().__init__(model, init_kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict_proba(X)[:, 1]

class GradientBoostingClassifier(sklearn.ensemble.GradientBoostingClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)

class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)

class DecisionTreeClassifier(sklearn.tree.DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)
        
class LogisticRegression(sklearn.linear_model.LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline(self, kwargs)



"""
from pipeline.gradient_boosting import GradientBoostingClassifier
import utils

files = utils.get_files("../data")
train = utils.csv.read(files["features_train_pre-tts.csv"]).dropna()
X = train.drop(["TARGET", "id_loan"], axis=1)
features = list(X.columns)
y = train["TARGET"].values.reshape(-1, 1)

to_woe = [
    "zipcode",
    "cd_msa",
    "st",
    "seller_name",
    "servicer_name",
    "prop_val_meth",
]

X_train, X_val, X_test, y_train, y_val, y_test = utils.train_test_split(X, y, 30000, val_size=10000, to_woe=to_woe, scaler="StandardScaler")
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

X_train, y_train = utils.undersampling.basic(X_train, y_train, ratio=1/10)
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


model = GradientBoostingClassifier()

model.pipeline.fit(X_train, y_train.reshape(-1))

y_hat = model.pipeline.predict(X_test.values)
model.pipeline.evaluate(y_test, y_hat)

model.pipeline.params

"""

