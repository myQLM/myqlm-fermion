# -*- coding: utf-8 -*-
"""
Custom gate sets
"""

import numpy as np

from qat.core import default_gate_set
from qat.lang.AQASM import AbstractGate


def mat_RXX(theta):
    t_cos = np.cos(theta / 2)
    t_sin = np.sin(theta / 2)
    mat = np.array([[t_cos, 0, 0, -1j * t_sin], [0, t_cos, -1j * t_sin, 0], [0, -1j * t_sin, t_cos, 0], [-1j * t_sin, 0, 0, t_cos]])
    return mat


def mat_RYY(theta):
    t_cos = np.cos(theta / 2)
    t_sin = np.sin(theta / 2)
    mat = np.array([[t_cos, 0, 0, +1j * t_sin], [0, t_cos, -1j * t_sin, 0], [0, -1j * t_sin, t_cos, 0], [+1j * t_sin, 0, 0, t_cos]])
    return mat


def mat_RZZ(theta):
    eminus = np.exp(-1j * theta / 2)
    eplus = np.exp(+1j * theta / 2)
    mat = np.array([[eminus, 0, 0, 0], [0, eplus, 0, 0], [0, 0, eplus, 0], [0, 0, 0, eminus]])
    return mat


def mat_CRXX(theta):
    mat_rxx = mat_RXX(theta)
    I4 = np.eye((4, 4), dtype=complex)
    mat = np.block([[I4, np.zeros((4, 4), dtype=complex)], [np.zeros((4, 4), dtype=complex), mat_rxx]])
    return mat


def mat_CRYY(theta):
    mat_ryy = mat_RYY(theta)
    I4 = np.eye((4, 4), dtype=complex)
    mat = np.block([[I4, np.zeros((4, 4), dtype=complex)], [np.zeros((4, 4), dtype=complex), mat_ryy]])
    return mat


def mat_CRZZ(theta):
    mat_rzz = mat_RZZ(theta)
    I4 = np.eye((4, 4), dtype=complex)
    mat = np.block([[I4, np.zeros((4, 4), dtype=complex)], [np.zeros((4, 4), dtype=complex), mat_rzz]])
    return mat


def get_custom_gate_set_1():
    mat_XX = np.array([[0, 1], [1, 0]])
    mat_XX = np.kron(mat_XX, mat_XX)
    mat_temp = np.zeros((2, 2), dtype=int)
    mat_temp[1][1] = 1
    mat_CXX = np.kron(mat_temp, mat_XX)
    for i in range(4):
        mat_CXX[i][i] = 1
    XX = AbstractGate("XX", [], arity=2, matrix_generator=lambda: mat_XX)
    CXX = AbstractGate("C-XX", [], arity=3, matrix_generator=lambda: mat_CXX)
    my_gate_set = default_gate_set()
    my_gate_set.add_signature(XX)
    my_gate_set.add_signature(CXX)

    # Multi qubits rotations
    RXX = AbstractGate("RXX", [float], arity=2, matrix_generator=lambda theta: mat_RXX(theta))
    RYY = AbstractGate("RYY", [float], arity=2, matrix_generator=lambda theta: mat_RYY(theta))
    RZZ = AbstractGate("RZZ", [float], arity=2, matrix_generator=lambda theta: mat_RZZ(theta))
    my_gate_set.add_signature(RXX)
    my_gate_set.add_signature(RYY)
    my_gate_set.add_signature(RZZ)

    return my_gate_set
