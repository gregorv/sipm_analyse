
import platform
major_version = int(platform.python_version_tuple()[0])
if major_version == 2:
    from py2 import *
elif major_version == 3:
    from .py3 import *


def import_raw_csv(filename):
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line:
                continue
            yield tuple(map(float, map(str.strip, line.split())))
