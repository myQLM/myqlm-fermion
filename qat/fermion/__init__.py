# -*- coding: utf-8 -*-

# Try to find other packages in other folders
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .hamiltonians import Hamiltonian, ElectronicStructureHamiltonian

# Backward compatiblity import
from .hamiltonians import SpinHamiltonian
from .hamiltonians import FermionHamiltonian

__all__ = [
    "Hamiltonian",
    "ElectronicStructureHamiltonian",
    "SpinHamiltonian",
    "FermionHamiltonian",
]
