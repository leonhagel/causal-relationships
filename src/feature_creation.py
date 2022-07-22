import utils
import numpy as np
import pandas as pd
from category_encoders import WOEEncoder
import sklearn.preprocessing

data = {
    "input": {
        "train": "../data/clean_train.csv", 
        "test": "../data/clean_test.csv",
    }, 
    "pre_tts": {
        "train": "../data/features_train_pre-tts_no-dummies.csv", 
        "test": "../data/features_test_pre-tts_no-dummies.csv",
    },
    "output": {
        "train": "../data/features_train_no-dummies.csv", 
        "test": "../data/features_test_no-dummies.csv",
    },
}

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

scaler = "StandardScaler"

def create_woe_encoding(to_woe, X_train, y_train, X_val=None, X_test=None, random_state=42, verbose=0):
    woe = WOEEncoder(cols = to_woe, random_state=random_state)
    out = [woe.fit_transform(X_train, y_train)]
    if not isinstance(X_val, type(None)):
        out += [woe.transform(X_val)]
    if not isinstance(X_test, type(None)):
        out += [woe.transform(X_test)]
    if len(out)==1:
        out = out[0]
    if verbose >= 1:
        print(f"finished creating woe encoding (encoded features: {to_woe}).")
    return out

## outlier treatment to be added

scalers = {
    "StandardScaler": sklearn.preprocessing.StandardScaler(),
    "MinMaxScaler": sklearn.preprocessing.MinMaxScaler()
}

def scale(scaler, X_train, X_val=None, X_test=None, verbose=0):
    out = [pd.DataFrame(scalers[scaler].fit_transform(X_train), columns=X_train.columns, index=X_train.index)]
    if not isinstance(X_val, type(None)): 
        out += [pd.DataFrame(scalers[scaler].transform(X_val), columns=X_val.columns, index=X_val.index)]
    if not isinstance(X_test, type(None)): 
        out += [pd.DataFrame(scalers[scaler].transform(X_test), columns=X_test.columns, index=X_test.index)]
    if len(out)==1:
        out = out[0]
    if verbose >= 1:
        print(f"finished scaling the data (used scaler: {scaler}).")    
    return out

verbose = 1
if __name__ == "__main__":
    train = utils.csv.read(data["input"]["train"], low_memory=False)
    test = utils.csv.read(data["input"]["test"])
    # dummy encoding
    train = pd.get_dummies(train, columns = to_dummy)
    test = pd.get_dummies(test, columns = to_dummy)
    missing_columns = [col for col in train.columns if all([col not in test.columns, col!="TARGET"])]
    test.loc[:, missing_columns] = 0
    test = test.loc[:, [col for col in train.columns if col != "TARGET"]]
    # dump pre train test split
    features = ["id_loan"] + [col for col in test.columns if col !="id_loan"]
    train = train.loc[:, features + ["TARGET"]]
    test = test.loc[:, features]
    utils.csv.dump(train, data["pre_tts"]["train"], verbose=verbose)
    utils.csv.dump(test, data["pre_tts"]["test"], verbose=verbose)
    # continuing without train test split
    id_loan = {
        "train": train["id_loan"],
        "test": test["id_loan"]
    }
    X_train = train.drop(['TARGET'], axis=1)
    y_train = train['TARGET']
    X_test = test
    # woe encoding
    X_train, X_test = create_woe_encoding(to_woe, X_train, y_train, X_test, verbose=verbose)
    # scaling 
    X_train, X_test = scale(scaler, X_train, X_test, verbose=verbose)
    # mean imputation
    X_test = X_test.fillna(X_train.mean())
    # dump
    train = X_train
    train["id_loan"] = id_loan["train"]
    X_test["id_loan"] = id_loan["test"]
    train["TARGET"] = y_train
    train = train.loc[:, features + ["TARGET"]]
    X_test = X_test.loc[:, features]
    utils.csv.dump(train, data["output"]["train"], verbose=verbose)
    utils.csv.dump(X_test, data["output"]["test"], verbose=verbose)