import sys
sys.path.append(".")
sys.path = [path for path in sys.path if "src/utils" not in path]

import argparse
import utils.json

filepath = "../setup/config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="key string for updated config entry", type=str)
    parser.add_argument("value", help="key string for updated config entry", type=str)
    args = parser.parse_args()
    try: 
        config = utils.json.load(filepath)
    except FileNotFoundError:
        config = {}
    config[str(args.key)] = str(args.value)
    utils.json.dump(config, filepath, sort_keys=True, verbose=1)
