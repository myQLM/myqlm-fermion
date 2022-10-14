# -*- coding: utf-8 -*-
"""
Integration test for VQE
"""

import random
import numpy as np
import unittest

from qat.lang.AQASM import QRoutine, RY, RX
from qat.qpus import get_default_qpu
from qat.vsolve.optimize.spsa_algorithm import spsa_minimize

from qat.fermion.transforms import transform_to_jw_basis
from qat.fermion.hamiltonians import ElectronicStructureHamiltonian
from qat.fermion.vqe import VQE


def simple_circuit(x):
    Qrout = QRoutine()
    Qrout.apply(RY(x[0]), 0)
    return Qrout


def simple_circuit_twoparameters(x):
    Qrout = QRoutine()
    Qrout.apply(RY(x[0]), 0)
    Qrout.apply(RX(x[1]), 1)
    return Qrout


class TestVQE(unittest.TestCase):
    def test_VQE_SPSA_JW_1qb(self):
        def spsa_optimizer(fun, theta0):
            return spsa_minimize(fun, theta0, precision=1e-6, maxiter=500, c=0.05, a=1, A=0)

        hpq = np.zeros((1, 1))
        hpq[0, 0] = 1
        hpqrs = np.zeros((1, 1, 1, 1))
        hamilt = ElectronicStructureHamiltonian(hpq=hpq, hpqrs=hpqrs)
        hamilt_sp = transform_to_jw_basis(hamilt)
        qpu = get_default_qpu()
        energy, _, _, _ = VQE(
            hamilt_sp,
            spsa_optimizer,
            simple_circuit,
            np.array([random.random() * 2 * np.pi]),
            qpu,
        )
        self.assertAlmostEqual(energy, 0.0, 3)

    def test_VQE_SPSA_JW_2qb(self):
        def spsa_optimizer(fun, theta0):
            return spsa_minimize(fun, theta0, precision=1e-6, maxiter=500, c=0.05, a=1, A=0)

        nqbit = 2
        hpq = np.zeros((nqbit, nqbit))
        hpq[0, 0] = hpq[1, 1] = -3
        hpq[1, 0] = hpq[0, 1] = 2
        hpqrs = np.zeros((nqbit, nqbit, nqbit, nqbit))

        hamiltonian = ElectronicStructureHamiltonian(hpq, hpqrs)
        min_theoritical_energy = min(np.linalg.eigvalsh(hamiltonian.get_matrix()))
        hamilt = ElectronicStructureHamiltonian(hpq=hpq, hpqrs=hpqrs)
        hamilt_sp = transform_to_jw_basis(hamilt)
        qpu = get_default_qpu()
        calculated_energy, _, _, _ = VQE(
            hamilt_sp,
            spsa_optimizer,
            simple_circuit_twoparameters,
            np.array([random.random() * 2 * np.pi, random.random() * 2 * np.pi]),
            qpu,
        )
        self.assertAlmostEqual(min_theoritical_energy, calculated_energy, delta=5e-3)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
