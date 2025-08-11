import importlib

def load(env_name):
    return importlib.import_module(f".{env_name}", package=__name__)
    