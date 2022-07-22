import json

def load(filepath):
    with open(filepath, "r") as file:
        config = json.load(file)
    return config

def dump(dictionary, filepath, indent=4, sort_keys=False, verbose=0, **kwargs):
    with open(filepath, "w") as file:
        json.dump(dictionary, file, indent=indent, sort_keys=sort_keys, **kwargs)
    if verbose >= 1:
        print(f"stored dictionary to {filepath}.")

def str2dict(string):
    return json.loads(string.replace("'", '"'))
