# -*- coding: utf-8 -*-

r"""
@authors Jean-Noel Quintin <jean-noel.quintin@atos.net>
@copyright 2017  Bull S.A.S.  -  All rights reserved.

           This is not Free or Open Source software.

           Please contact Bull SAS for details about its license.

           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

"""

# Try to find other packages in other folders
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .hamiltonians import Hamiltonian, ElectronicStructureHamiltonian

# Backward compatiblity import
from .hamiltonians import SpinHamiltonian
from .hamiltonians import FermionHamiltonian

from .ansatz_generator import AnsatzGenerator
from .observable_generator import ObservableGenerator

__all__ = ["AnsatzGenerator", "ObservableGenerator"]
