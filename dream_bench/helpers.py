import sys
from click import secho
from importlib import import_module


def is_url(x: str):
    if x.startswith("http://") or x.startswith("https://"):
        return True

    return False


def exists(x):
    return x is not None


def filename_from_path(path):
    return path.split("/")[-1]


def import_or_print_error(pkg_name, err_str=None, **kwargs):
    try:
        return import_module(pkg_name)
    except ModuleNotFoundError as _:
        if exists(err_str):
            secho(err_str, **kwargs)
        return sys.exit()


def print_ribbon(s, symbol="=", repeat=10):
    flank = symbol * repeat
    return f"{flank} {s} {flank}"
