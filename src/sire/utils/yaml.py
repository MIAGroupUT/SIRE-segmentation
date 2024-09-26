import yaml


def load_config(config_path: str):
    with open(config_path, "r") as stream:
        d = yaml.safe_load(stream)

    return d
