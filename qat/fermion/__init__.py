# -*- coding: utf-8 -*-

# Try to find other packages in other folders
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .hamiltonians import Hamiltonian, ElectronicStructureHamiltonian
# from .ansatz_generator import AnsatzGenerator
# from .observable_generator import ObservableGenerator

# from .ansatz_generator import AnsatzGenerator
# from .observable_generator import ObservableGenerator

# Backward compatiblity import
from .hamiltonians import SpinHamiltonian
from .hamiltonians import FermionHamiltonian

<<<<<<< HEAD
__all__ = ["Hamiltonian", "ElectronicStructureHamiltonian"]#, "AnsatzGenerator", "ObservableGenerator"]
=======
__all__ = ["Hamiltonian", "ElectronicStructureHamiltonian"]  # , "AnsatzGenerator", "ObservableGenerator"]
>>>>>>> 4f4ee26... update version + notebooks + tests + fix dependencies
