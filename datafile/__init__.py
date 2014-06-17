
import platform
major_version = int(platform.python_version_tuple()[0])
if major_version == 2:
    from py2 import *
elif major_version == 3:
    from .py3 import *


def import_raw_csv(filename):
    with open(filename) as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            yield tuple(map(float, map(str.strip, line.split())))


def import_csv_header(filename):
    """
    All comments at the beginning of the files
    are considered the header of the file.

    Empty lines do not break the header!

    Only non-empty, non-comment lines end the header.
    """
    with open(filename) as f:
        for line in f:
            if not line.strip():
                continue
            if not line.strip().startswith('#'):
                break
            yield line[1:].strip()


def userheader2dict(header):
    dheader = {}
    for line in header:
        try:
            key, val = map(str.strip, line.split("="))
            if key.endswith("_f"):
                val = float(val)
            elif key.endswith("_b"):
                val = (val.lower() == "true")
            dheader[key] = val
        except Exception:
            pass
    return dheader


