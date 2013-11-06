
import platform
major_version = int(platform.python_version_tuple()[0])
if major_version == 2:
    from py2 import *
elif major_version == 3:
    from .py3 import *