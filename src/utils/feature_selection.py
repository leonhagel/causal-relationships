import utils

def get_markov_blanket(M, y="TARGET", threshold=0, filepath=None, verbose=0):
    children = list(M.columns[M.loc[y, :] > threshold])
    parents = list(M.columns[M.loc[:, y] > threshold])
    spouses = list(M.columns[M.loc[:, children].sum(axis=1) > threshold])
    for ls in [parents, spouses, children]:
        try:
            ls.pop(ls.index(y))
        except ValueError:
            pass
    markov_blanket = {"y": [y], "parents": parents, "spouses": spouses, "children": children}
    if verbose >= 1:
        indent = max([len(key) for key in markov_blanket.keys()]) + 4
        print("markov blanket:")
        for key, features in markov_blanket.items():
            print(f"{key}:{' '*(indent-len(key))}{features}")
    if not isinstance(filepath, type(None)):
        utils.json.dump(markov_blanket, filepath, verbose=verbose)
    return markov_blanket

def get_causal_features(M=None, markov_blanket=None, y="TARGET", threshold=0):
    if isinstance(markov_blanket, type(None)):
        markov_blanket = get_markov_blanket(M)
    features = list(set(sum(markov_blanket.values(), [])))
    features.pop(features.index(y))
    return features
