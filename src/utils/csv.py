import pandas as pd
from utils.json import load

def read(filepath, dtype=None, **kwargs):
    if isinstance(dtype, str):
        dtype = load(dtype)
    return pd.read_csv(filepath, dtype=dtype, **kwargs)

def dump(df, filepath, index=False, verbose=1, **kwargs):
    df.to_csv(filepath, index=index, **kwargs)
    if verbose>=1:
        print(f"stored dataframe to {filepath}.")
