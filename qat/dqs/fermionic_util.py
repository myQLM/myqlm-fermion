"""
Functions to do exact computations on fermionic Hamiltonians
"""
import copy
import scipy.linalg as la
import numpy as np

from .hamiltonians import ElectronicStructureHamiltonian
from .util import init_creation_ops, dag


def exact_eigen_energies(hpq, hpqrs):
    r"""
    This function returns eigenvalues and vector of an Hamiltonian defined by

    .. math::

        H = \sum_{pq} h_{pq}a_p^\dagger a_q
          + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s

    Args:
        hpq (list): 2D list
        hpqrs (list): 4D list

    Returns:
        (numpy.ndarray, numpy.ndarray): eigenenergy, eigenvectors (as column vectors, i.e. for eigenenergy i of the array, the corresponding vector will be [:, i])
    """
    H = ElectronicStructureHamiltonian(hpq, hpqrs).get_matrix()
    E, eigvecs = np.linalg.eigh(H)
    return E, eigvecs


def fermionic_hamiltonian_exponential(hpq, hpqrs):
    r"""
    This function return the matricial expression of
    :math:`e^{-i h_{pq} a^\dagger_p a_q + h.c.}` or
    :math:`e^{-i h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s +h.c.}`.

    Args:
        hpq (list): 2D list
        hpqrs (list): 4D list

    Returns:
        numpy.ndarray
    """
    hamilt = ElectronicStructureHamiltonian(hpq, hpqrs)
    matrix = hamilt.get_matrix()
    return la.expm(-1j * matrix)


def make_intermediate_hamiltonian(alpha, hamiltonian):
    r"""
    This function returns

    .. math::
        H_\alpha = (1-\alpha) H_{dens} + \alpha H

    with H_dens: the part of H which commutes with the density operator


    Args:
        alpha (float): the parameter which corresponds to :math:`H_\alpha = \alpha H_{FCI} + (1-\alpha) H_{HF}`
        hamiltonian (ElectronicStructureHamiltonian): the original Hamiltonian

    Returns:
         ElectronicStructureHamiltonian
    """
    hpq_FCI = hamiltonian.hpq
    hpqrs_FCI = hamiltonian.hpqrs
    number_qubits = len(hpqrs_FCI)
    hpq_HF = np.diag(np.diag(hpq_FCI))
    hpq = copy.deepcopy(hpq_FCI)
    np.fill_diagonal(hpq, 0)
    hpqrs_HF = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs = copy.deepcopy(hpqrs_FCI)
    for p in range(len(hpqrs_FCI)):
        for q in range(p):
            hpqrs_HF[p][q][q][p] = hpqrs_FCI[p][q][q][p]
            hpqrs_HF[q][p][q][p] = hpqrs_FCI[q][p][q][p]
            hpqrs_HF[p][q][p][q] = hpqrs_FCI[p][q][p][q]
            hpqrs_HF[q][p][p][q] = hpqrs_FCI[q][p][p][q]
            hpqrs[p][q][q][p] = hpqrs[q][p][q][p] = hpqrs[p][q][p][q] = hpqrs[q][p][p][q] = 0

    return ElectronicStructureHamiltonian(hpq_HF + (alpha) * hpq, hpqrs_HF + (alpha) * hpqrs)
