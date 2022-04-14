#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing Hamiltonians
"""
import numpy as np

from qat.core import Observable, Term
from qat.fermion.hamiltonians import Hamiltonian, ObservableType


def test_spin_hamiltonian():
    obs = Observable(
        2,
        pauli_terms=[
            Term(0.5, "X", [0]),
            Term(0.25, "Y", [1]),
            Term(1.0, "ZZ", [0, 1]),
        ],
    )

    # convert to Hamiltonian
    spin_hamiltonian = Hamiltonian(obs.nbqbits, obs.terms, obs.constant_coeff)

    # get matrix
    spin_h_matrix = spin_hamiltonian.get_matrix()

    # diagonalize
    eigvals, eigvecs = np.linalg.eigh(spin_h_matrix)

    print("Eigenvalues=", eigvals)
    np.testing.assert_almost_equal(min(eigvals), -1.25, decimal=13)


def test_to_spin_transform():

    hamiltonian = Hamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
    h_matrix = hamiltonian.get_matrix()

    spin_h = hamiltonian.to_spin()
    spin_matrix = spin_h.get_matrix()

    assert spin_h.htype == ObservableType.SPIN
    np.testing.assert_equal(h_matrix, spin_matrix)
