# -*- coding: utf-8 -*-
"""
Unitary tests for utility functions
"""

import numpy as np
from qat.lang.AQASM import QRoutine, X, Y, H, Z, CNOT

from qat.fermion.util import get_unitary_from_circuit

H_matrix = np.array([[1, 1], [1, -1]]) / (2**0.5)
CNOT_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
Id_matrix = np.eye(2, 2)
Sz_matrix = np.array([[1, 0], [0, -1]])
Sx_matrix = np.array([[0, 1], [1, 0]])
Sy_matrix = np.array([[0, -1j], [1j, 0]])


def test_get_unitary_1():
    q_rout = QRoutine()
    q_rout.apply(Y, 0)
    q_rout.apply(X, 1)
    q_rout.apply(H, 0)
    a = get_unitary_from_circuit(q_rout, 2)
    b = np.kron(np.dot(H_matrix, Sy_matrix), Sx_matrix)
    np.testing.assert_almost_equal(np.linalg.norm(a - b), 0, decimal=4)


def test_get_unitary_2():
    q_rout = QRoutine()
    q_rout.apply(CNOT, 0, 1)
    q_rout.apply(CNOT, 1, 2)
    a = get_unitary_from_circuit(q_rout, 3)
    b = np.dot(np.kron(Id_matrix, CNOT_matrix), np.kron(CNOT_matrix, Id_matrix))
    np.testing.assert_almost_equal(np.linalg.norm(a - b), 0, decimal=4)


def test_get_unitary_3():
    q_rout = QRoutine()
    q_rout.apply(X, 0)
    q_rout.apply(X, 1)
    q_rout.apply(H, 0)
    q_rout.apply(H, 2)
    q_rout.apply(Z, 2)
    q_rout.apply(CNOT, [1, 2])
    a = get_unitary_from_circuit(q_rout, 3)
    b = np.dot(
        np.kron(Id_matrix, CNOT_matrix),
        np.dot(
            np.kron(Id_matrix, np.kron(Id_matrix, Sz_matrix)),
            np.dot(
                np.kron(Id_matrix, np.kron(Id_matrix, H_matrix)),
                np.dot(
                    np.kron(H_matrix, np.kron(Id_matrix, Id_matrix)),
                    np.dot(
                        np.kron(Id_matrix, np.kron(Sx_matrix, Id_matrix)),
                        np.kron(Sx_matrix, np.kron(Id_matrix, Id_matrix)),
                    ),
                ),
            ),
        ),
    )
    np.testing.assert_almost_equal(np.linalg.norm(a - b), 0, decimal=4)
