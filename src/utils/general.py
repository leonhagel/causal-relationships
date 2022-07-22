import os

def get_files(path):
    """returns a dictionary filename: filepath for a given path."""
    files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filename in files.keys():
                filename = filepath.replace(os.path.join(path, ""), "")
            files[filename] = filepath
    return files


def str2ratio(string):
    if string != "None":
        split = [float(i) for i in string.split("/")]
        out = split[0]/split[1]
    else:
        out = None
    return out
