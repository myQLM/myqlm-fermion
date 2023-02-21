# -*- coding: utf-8 -*-
"""
Init
"""

# Try to find other packages in other folders (with separate build directory)
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Initialize dynamic modules
from .modules import register_modules
register_modules()

