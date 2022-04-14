# -*- coding: utf-8 -*-

r"""
@authors Jean-Noel Quintin <jean-noel.quintin@atos.net>
@copyright 2017  Bull S.A.S.  -  All rights reserved.

           This is not Free or Open Source software.

           Please contact Bull SAS for details about its license.

           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

"""
from .wrapper import MoleculeInfo, MolecularHamiltonian

# Try to find other packages in other folders (with separate build directory)
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .wrapper import MolecularHamiltonian, MoleculeInfo
from .ucc import guess_init_params, get_cluster_ops, get_hf_ket, get_active_space_hamiltonian, select_active_orbitals
from .qse import apply_quantum_subspace_expansion
from .pyscf_tools import perform_pyscf_computation
