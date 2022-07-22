import sklearn.model_selection
from feature_creation import create_woe_encoding, scale

def train_test_split(X, y, test_size=None, val_size=None, to_woe=None, scaler=None, random_state=42, train_index=None, test_index=None, **kwargs):
    if all([isinstance(train_index, type(None)), isinstance(test_index, type(None))]):
        # standard train test split
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, **kwargs)
        if not isinstance(val_size, type(None)):
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=val_size, **kwargs)
        else:
            X_val = None
    else:
        # train test split for k fold
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        X_val = None
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        y_val = None
        
    if not isinstance(to_woe, type(None)):
        temp = create_woe_encoding(to_woe, X_train, y_train, X_val, X_test, random_state=random_state)
        if len(temp)==3:
            X_train, X_val, X_test = temp
        else:
            X_train, X_test = temp
    if not isinstance(scaler, type(None)):
        temp = scale(scaler, X_train, X_val, X_test)
        if len(temp)==3:
            X_train, X_val, X_test = temp
        else:
            X_train, X_test = temp
    if not isinstance(X_val, type(None)):
        out = X_train, X_val, X_test, y_train, y_val, y_test
    else:
        out = X_train, X_test, y_train, y_test
    return out