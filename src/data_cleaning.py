import utils
import pandas as pd

data = {
    "../data/data_train.csv": "../data/clean_train.csv",
    "../data/x_test.csv": "../data/clean_test.csv"
}

# missing values
missing = {
    "fico": {9999: None},
    #"flag_fthb": {"9": None}, might contain some information
    "cd_msa": {"     ": None},
    "mi_pct": {999: None},
    "cnt_units": {99: None},
    "occpy_sts": {"9": None},
    "cltv": {999: None},
    "dti": {999: None},
    "ltv": {999: None},
    "channel": {"9": None},
    "prop_type": {"99": None},
    "zipcode": {"     ": None},
    "loan_purpose": {"9": None},
    "cnt_borr": {"99": None},
    # "pgrm_ind": {"9": None},  might contain some information
    "prop_val_meth": {"9": None}
}

# binary features 
binary = {
    "flag_fthb": {"Y": 1, "N": 0},
    "ppmt_pnlty": {"Y": 1, "N": 0},
    "flag_sc": {"Y": 1, None: 0},
    "rel_ref_ind": {"Y": 1, None: 0},
    "int_only_ind": {"Y": 1, "N": 0}    
}

# drop features
drop_columns = ['dt_first_pi', 'dt_matr', 'pre_relief']

def replace(df, translation, verbose=0):
    """
    translation: dict
    - {"column_name": {old_value: new_value}}
    """
    out = df.copy()
    if any(sum([[old == None for old in mapping.keys()] for mapping in translation.values()], [])):
        replace = {column_name: {old: new for old, new in mapping.items() if old != None} for column_name, mapping in translation.items()}
        fillna = {column_name: mapping[None] for column_name, mapping in translation.items() if any([old == None for old in mapping.keys()])}
        out = out.fillna(fillna)
        
    else:
        replace = translation
    out = out.replace(replace)
    if verbose >= 1:
        indent = max([len(name) for name in translation.keys()]) + 4
        for column_name, mapping in translation.items():
            old_values = list(mapping.keys())
            n_filled_na = df[column_name].isna().sum() if any([old == None for old in mapping.keys()]) else 0
            print(f"{column_name}:{' '*(indent-len(column_name))}replaced {df[column_name].isin(old_values).sum() + n_filled_na} values ({', '.join([f'{k} -> {v}' for k, v in mapping.items()])}).")
    return out

verbose = 1
if __name__ == "__main__":
    for i, o in data.items():
        print(f"starting to clean {i}...")
        df = utils.csv.read(i, low_memory=False)
        df = replace(df, missing, verbose=verbose)
        df = replace(df, binary, verbose=verbose)
        df = df.drop(drop_columns, axis=1)
        utils.csv.dump(df, o, verbose=verbose)
        
    