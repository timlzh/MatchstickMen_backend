import os

debug = os.environ.get("DEBUG", default=1)


def dprint(str):
    if debug is not None:
        print(str)
