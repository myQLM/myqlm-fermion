#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a container for various ansatz circuits for variational preparation of fermionic states.
"""

import numpy as np

from qat.lang.AQASM import Program, RY, CNOT, X

from qat.fermion.matchgates import nb_params_LDCA, LDCA_routine
from qat.fermion.util import make_fSim_fan_routine, make_sugisaki_routine, tobin

from qat.pbo import GraphCircuit, VAR
from qat.lang.AQASM import Program, H, RX, RY, RZ, CNOT


def make_trotter_slice(op, iter_num):
    r"""
    Make Trotter slice corresponding to \exp(-i \theta_i O)

    Args:
        op (Observable): operator O (Hermitian)
        iter_num (int): index i of parameter theta_i

    Returns:
        Program: the corresponding program
    """
    prog = Program()
    var = prog.new_var(float, "theta_" + str(iter_num))
    qbits = prog.qalloc(op.nbqbits)

    if len(op.terms) == 1 and len(op.terms[0].qbits) == 1:
        term = op.terms[0]
        pauli_string = term.op
        coeff = term.coeff.real
        pos = term.qbits[0]
        if pauli_string == "X":
            prog.apply(RX(2 * coeff * var), qbits[pos])
        elif pauli_string == "Y":
            prog.apply(RY(2 * coeff * var), qbits[pos])
        elif pauli_string == "Z":
            prog.apply(RZ(2 * coeff * var), qbits[pos])
        return prog

    for i in range(len(op.terms)):
        pauli_string = op.terms[i].op
        list_qbits = op.terms[i].qbits
        coeff = op.terms[i].coeff.real
        # add RX(np.pi/2) for Y-gates and H for X-gates
        for current_pauli_op, current_qbit in zip(pauli_string, list_qbits):
            if current_pauli_op == "Y":
                prog.apply(RX(np.pi / 2), qbits[current_qbit])
            elif current_pauli_op == "X":
                prog.apply(H, qbits[current_qbit])
        # add CNOT gates
        for j in range(len(pauli_string) - 1):
            current_qbit = list_qbits[j]
            next_qbit = list_qbits[j + 1]
            prog.apply(CNOT, qbits[current_qbit], qbits[next_qbit])
        # add RZ-gate
        prog.apply(RZ(2 * coeff * var), qbits[next_qbit])
        # add CNOT gates back
        for j in range(len(pauli_string) - 1, 0, -1):
            current_qbit = list_qbits[j]
            previous_qbit = list_qbits[j - 1]
            prog.apply(CNOT, qbits[previous_qbit], qbits[current_qbit])
        # add RX(-np.pi/2) for Y-gates and H for X-gates back
        for current_pauli_op, current_qbit in zip(pauli_string, list_qbits):
            if current_pauli_op == "Y":
                prog.apply(RX(-np.pi / 2), qbits[current_qbit])
            elif current_pauli_op == "X":
                prog.apply(H, qbits[current_qbit])
    return prog


def make_ldca_circ(nb_fermionic_modes, ncycles, eigstate_ind=0, slater=False):
    """
    Construct a LDCA circuit, applying ncycles layers of matchgates routines
    on nb_fermionic_modes qubits.

    Args:
        nb_fermionic_modes (int): number of qubits
        ncycles (int): number of LDCA cycles
        eigstate_ind (int, optional): defaults to 0
        slater (bool, optional): whether to only include excitation-preserving rotations.
                                 Defaults to False.

    Return:
       :class:`~qat.core.Circuit`
    """

    prog = Program()
    nqbits = nb_fermionic_modes
    reg = prog.qalloc(nqbits)

    nstring = tobin(eigstate_ind, nqbits)
    if slater:
        nonzerobits = [index for index in range(nqbits) if nstring[index] == "1"]
    else:
        nonzerobits = list(range(nqbits))
    for b in nonzerobits:
        prog.apply(X, reg[b])

    nb_params = nb_params_LDCA(nqbits, ncycles, slater=slater)
    theta = [prog.new_var(float, r"\theta_{%i}" % i) for i in range(nb_params)]
    prog.apply(LDCA_routine(nqbits, ncycles, theta, None, slater=slater), reg)

    return prog.to_circ()


def make_mr_circ():
    """
    Builds a small, one-parameter Multi-Reference (MR) circuit on 4 qubits inspired from Sugisaki et al.,
    10.1021/acscentsci.8b00788 [2019] to prepare states in natural orbitals.

    Returns:
        :class:`~qat.core.Circuit`
    """

    prog = Program()
    reg = prog.qalloc(4)
    theta = [prog.new_var(float, r"\theta_{%i}" % i) for i in range(1)]
    prog.apply(X, reg[2])
    prog.apply(X, reg[3])
    prog.apply(RY(theta[0]), reg[0])
    prog.apply(CNOT, reg[0], reg[1])
    prog.apply(CNOT, reg[0], reg[2])
    prog.apply(CNOT, reg[1], reg[3])

    circ = prog.to_circ()

    return circ


def make_mrep_circ(n_fSim_cycles=4, set_phi_to_0=False):
    """
    Constructs the 8-qubit Multi-Reference Excitation Preserving (MREP) ansatz that combines
    the multi-reference routine of Sugisaki et al., 10.1021/acscentsci.8b00788 [2019] with some fSim nearest-neighbour cycles.
    The second angles of the fSim gates (phi) may be taken to 0.

    Args:
        n_fSim_cycles (int, optional): number of fSim cycles, defaults to 4.
        set_phi_to_0 (bool, optional): whether to set all second angles in the fSim gates to 0 (True)
                                       or not (False). Defaults to False.
    Returns:
        :class:`~qat.core.Circuit`
    """
    nbqbits = 8

    prog = Program()
    reg = prog.qalloc(nbqbits)
    theta = [
        prog.new_var(float, "\\theta_{%i}" % i) for i in range(2 + 14 * n_fSim_cycles)
    ]

    for i in range(nbqbits // 2, nbqbits):
        prog.apply(X, reg[i])

    rout = make_sugisaki_routine(theta[0])
    prog.apply(rout, reg[2], reg[3], reg[4], reg[5])
    rout = make_sugisaki_routine(theta[1])
    prog.apply(rout, reg[0], reg[1], reg[6], reg[7])

    ind = 2

    for _ in range(n_fSim_cycles):
        fSim_angles = theta[ind : (ind + 14)]
        if set_phi_to_0:
            for i in range(len(fSim_angles) // 2):
                fSim_angles[2 * i + 1] = 0
        rout = make_fSim_fan_routine(nbqbits, fSim_angles)
        prog.apply(rout, reg)
        ind += 14

    circ = prog.to_circ()

    return circ


def make_shallow_circ():
    """
    Builds the 8-parameter circuit proposed in Keen et al., 10.1088/2058-9565/ab7d4c (arXiv:1910.09512) [2019].
    This is a 4-qubit circuit.

    Returns:
        :class:`~qat.core.Circuit`
    """

    prog = Program()
    reg = prog.qalloc(4)
    theta = [prog.new_var(float, r"\theta_{%i}" % i) for i in range(8)]

    for i in range(4):
        prog.apply(RY(theta[i]), reg[i])

    prog.apply(CNOT, reg[2], reg[3])
    prog.apply(RY(theta[4]), reg[2])
    prog.apply(RY(theta[5]), reg[3])
    prog.apply(CNOT, reg[0], reg[2])
    prog.apply(RY(theta[6]), reg[0])
    prog.apply(RY(theta[7]), reg[2])
    prog.apply(CNOT, reg[0], reg[1])

    return prog.to_circ()


def make_general_hwe_circ(
    nqbits, n_cycles=1, rotation_gates=[RY], entangling_gate=CNOT
):
    """
    Constructs an ansatz made of :math:`n_{\mathrm{cycles}}` layers of so-called thinly-dressed routines,
    that is to say entanglers surrounded by four one-qubit rotations are applied on nearest-neighbour
    qubits in an odd/even alternating pattern.

    This circuit is typically of the hardware-efficient class.

    Args:
        nqbits (int): number of qubits of the circuit
        n_cycles (int): number of layers
        rotation_gates (list of :class:`~qat.lang.AQASM.gates.Gate` objects with arity 1):
                        parametrized rotation gates to include around the
                        entangling gate, defaults to :math:`RY`
        entangling_gate (:class:`~qat.lang.AQASM.gates.Gate`, arity=2): the 2-qubit
                                                        entangler, defaults to :math:`CNOT`

    Returns:
        :class:`~qat.core.Circuit`
    """
    n_rotations = len(rotation_gates)

    prog = Program()
    reg = prog.qalloc(nqbits)
    theta = [
        prog.new_var(float, r"\theta_{%i}" % i)
        for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))
    ]
    ind_theta = 0

    for i in range(nqbits):
        for R in rotation_gates:
            prog.apply(R(theta[ind_theta]), reg[i])
            ind_theta += 1

    for _ in range(n_cycles):
        for i in range(nqbits // 2):
            prog.apply(entangling_gate, reg[2 * i], reg[2 * i + 1])
            for R in rotation_gates:
                prog.apply(R(theta[ind_theta]), reg[2 * i])
                ind_theta += 1
                prog.apply(R(theta[ind_theta]), reg[2 * i + 1])
                ind_theta += 1

        for i in range(nqbits // 2 - 1):
            prog.apply(entangling_gate, reg[2 * i + 1], reg[2 * i + 2])
            for R in rotation_gates:
                prog.apply(R(theta[ind_theta]), reg[2 * i + 1])
                ind_theta += 1
                prog.apply(R(theta[ind_theta]), reg[2 * i + 2])
                ind_theta += 1

    return prog.to_circ()


def make_compressed_ldca_circ(
    nb_fermionic_modes, ncycles, eigstate_ind=0, slater=False
):
    """
    Builds a compressed version of the LDCA ansatz circuit.

    The new pattern was obtained using qat.synthopline.

    Args:
        nb_fermionic_modes (int): number of qubits
        ncycles (int): number of LDCA cycles
        eigstate_ind (int, optional): defaults to 0
        slater (bool, optional): whether to only include excitation-preserving rotations.
                                 Defaults to False.

    Return:
       :class:`~qat.core.Circuit`
    """

    circ = make_ldca_circ(
        nb_fermionic_modes, ncycles, eigstate_ind=eigstate_ind, slater=slater
    )

    graph = GraphCircuit()
    graph.load_circuit(circ)

    a1 = VAR()
    a2 = VAR()
    a3 = VAR()
    a4 = VAR()
    a5 = VAR()

    old_pattern = [
        ("RYY", [0, 1], a3),
        ("RXX", [0, 1], a2),
        ("RZZ", [0, 1], a1),
        ("RYX", [0, 1], a5),
        ("RXY", [0, 1], a4),
    ]
    new_pattern = [
        ("CNOT", [1, 0]),
        ("RX", [1], a2),
        ("RY", [1], a4),
        ("H", [1]),
        ("CNOT", [0, 1]),
        ("PH", [1], -a3),
        ("PH", [0], a1),
        ("RY", [1], -a5),
        ("CNOT", [1, 0]),
        ("H", [1]),
        ("CNOT", [0, 1]),
    ]

    # Replace pattern
    while graph.replace_pattern(old_pattern, new_pattern):
        continue

    compressed_circ = graph.to_circ()

    return compressed_circ