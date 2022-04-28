# -*- coding: utf-8 -*-

# Try to find other packages in other folders (with separate build directory)
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
