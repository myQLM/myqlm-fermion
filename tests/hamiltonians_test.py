
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing Hamiltonians
"""
import unittest
from itertools import product
import numpy as np

from qat.core import Observable, Term
from qat.dqs import SpinHamiltonian


class TestSpinHamiltonian(unittest.TestCase):
    def test_spin_hamiltonian(self):
        obs = Observable(2, pauli_terms=[Term(0.5, "X", [0]),
                                         Term(0.25, "Y", [1]),
                                         Term(1.0, "ZZ", [0, 1])])


        # convert to SpinHamiltonian
        spin_hamiltonian = SpinHamiltonian(obs.nbqbits, obs.terms, obs.constant_coeff)

        # get matrix
        spin_h_matrix = spin_hamiltonian.get_matrix()

        # diagonalize
        eigvals, eigvecs = np.linalg.eigh(spin_h_matrix)

        print("Eigenvalues=", eigvals)
        self.assertAlmostEqual(min(eigvals), -1.25, delta=1e-13)


