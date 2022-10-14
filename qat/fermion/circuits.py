#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Container for various ansatz circuits for variational preparation of fermionic states.
"""

from typing import Optional, List
import numpy as np
import warnings

from qat.core import Observable, Circuit
from qat.lang.AQASM import Program, CNOT, X, H, RX, RY, RZ
from qat.lang.AQASM.gates import Gate

from .matchgates import nb_params_LDCA, LDCA_routine
from .util import make_fSim_fan_routine, make_sugisaki_routine, tobin


def make_trotter_slice(op: Observable, iter_num: int) -> Program:
    r"""
    Make Trotter slice corresponding to \exp(-i \theta_i O).

    Args:
        op (Observable): Operator O (Hermitian).
        iter_num (int): Index i of parameter theta_i.

    Returns:
        Program: The corresponding program
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

    else:

        for i, _ in enumerate(op.terms):

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


def make_ldca_circ(
    nb_fermionic_modes: int,
    ncycles: int,
    eigstate_ind: Optional[int] = 0,
    slater: Optional[bool] = False,
) -> Circuit:
    r"""
    Construct a LDCA circuit (see `article by P. Dallaire-Demers et al. (2019) <https://doi.org/10.48550/arXiv.1801.01053>`_),
    applying ncycles layers of matchgates routines on nb_fermionic_modes qubits.

    Args:
        nb_fermionic_modes (int): Number of qubits.
        ncycles (int): Number of LDCA cycles.
        eigstate_ind (int, optional): Eigenstate index. Defaults to 0.
        slater (Optional[bool]): Whether to only include excitation-preserving rotations. Defaults to False.

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
    for bit in nonzerobits:
        prog.apply(X, reg[bit])

    nb_params = nb_params_LDCA(nqbits, ncycles, slater=slater)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(nb_params)]
    prog.apply(LDCA_routine(nqbits, ncycles, theta, None, slater=slater), reg)

    return prog.to_circ()


def make_mr_circ() -> Circuit:
    """
    Builds a small, one-parameter Multi-Reference (MR) circuit on 4 qubits inspired from `Sugisaki et al. article (2019)
    <https://doi.org/10.1021/acscentsci.8b00788>`_ to prepare states in natural orbitals.

    Returns:
        :class:`~qat.core.Circuit`

    """

    prog = Program()
    reg = prog.qalloc(4)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(1)]
    prog.apply(X, reg[2])
    prog.apply(X, reg[3])
    prog.apply(RY(theta[0]), reg[0])
    prog.apply(CNOT, reg[0], reg[1])
    prog.apply(CNOT, reg[0], reg[2])
    prog.apply(CNOT, reg[1], reg[3])

    circ = prog.to_circ()

    return circ


def make_mrep_circ(n_fsim_cycles: Optional[int] = 4, set_phi_to_0: Optional[bool] = False) -> Circuit:
    """
    Constructs the 8-qubit Multi-Reference Excitation Preserving (MREP) ansatz that combines
    the multi-reference routine of `Sugisaki et al. article (2019) <https://doi.org/10.1021/acscentsci.8b00788>`_ with some fSim
    nearest-neighbour cycles.
    The second angles of the fSim gates (phi) may be taken to 0.

    Args:
        n_fsim_cycles (int, optional): Number of fSim cycles, defaults to 4.
        set_phi_to_0 (bool, optional): Whether to set all second angles in the fSim gates to 0 (True)
                                       or not (False). Defaults to False.
    Returns:
        :class:`~qat.core.Circuit`

    """
    nbqbits = 8

    prog = Program()
    reg = prog.qalloc(nbqbits)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(2 + 14 * n_fsim_cycles)]

    for i in range(nbqbits // 2, nbqbits):
        prog.apply(X, reg[i])

    rout = make_sugisaki_routine(theta[0])
    prog.apply(rout, reg[2], reg[3], reg[4], reg[5])
    rout = make_sugisaki_routine(theta[1])
    prog.apply(rout, reg[0], reg[1], reg[6], reg[7])

    ind = 2

    for _ in range(n_fsim_cycles):
        fsim_angles = theta[ind : (ind + 14)]
        if set_phi_to_0:
            for i in range(len(fsim_angles) // 2):
                fsim_angles[2 * i + 1] = 0
        rout = make_fSim_fan_routine(nbqbits, fsim_angles)
        prog.apply(rout, reg)
        ind += 14

    circ = prog.to_circ()

    return circ


def make_shallow_circ() -> Circuit:
    """
    Builds the 8-parameter circuit proposed in `Keen et al. article (2019) <https://doi.org/10.48550/arXiv.1910.09512>`_.
    This is a 4-qubit circuit.

    Returns:
        :class:`~qat.core.Circuit`

    """

    prog = Program()
    reg = prog.qalloc(4)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(8)]

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
    nqbits: int,
    n_cycles: int = 1,
    rotation_gates: List[Gate] = None,
    entangling_gate: Gate = CNOT,
) -> Circuit:
    r"""
    Constructs an ansatz made of :math:`n_{\mathrm{cycles}}` layers of so-called thinly-dressed routines,
    that is to say entanglers surrounded by four one-qubit rotations are applied on nearest-neighbour
    qubits in an odd/even alternating pattern.

    This circuit is typically of the hardware-efficient class.

    Args:
        nqbits (int): Number of qubits of the circuit.
        n_cycles (int): Number of layers.
        rotation_gates (List[Gate]): Parametrized rotation gates to include around the entangling gate. Defaults to :math:`RY`. Must
            be of arity 1.
        entangling_gate (Gate): The 2-qubit entangler. Must be of arity 2. Defaults to :math:`CNOT`.

    Returns:
        :class:`~qat.core.Circuit`

    """
    if rotation_gates is None:
        rotation_gates = [RY]

    n_rotations = len(rotation_gates)

    prog = Program()
    reg = prog.qalloc(nqbits)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    ind_theta = 0

    for i in range(nqbits):

        for rot in rotation_gates:

            prog.apply(rot(theta[ind_theta]), reg[i])
            ind_theta += 1

    for _ in range(n_cycles):

        for i in range(nqbits // 2):

            prog.apply(entangling_gate, reg[2 * i], reg[2 * i + 1])

            for rot in rotation_gates:

                prog.apply(rot(theta[ind_theta]), reg[2 * i])
                ind_theta += 1
                prog.apply(rot(theta[ind_theta]), reg[2 * i + 1])
                ind_theta += 1

        for i in range(nqbits // 2 - 1):

            prog.apply(entangling_gate, reg[2 * i + 1], reg[2 * i + 2])

            for rot in rotation_gates:

                prog.apply(rot(theta[ind_theta]), reg[2 * i + 1])
                ind_theta += 1
                prog.apply(rot(theta[ind_theta]), reg[2 * i + 2])
                ind_theta += 1

    return prog.to_circ()


def make_compressed_ldca_circ(
    nb_fermionic_modes: int,
    ncycles: int,
    eigstate_ind: Optional[int] = 0,
    slater: Optional[bool] = False,
) -> Circuit:
    """
    Builds a compressed version of the LDCA ansatz circuit.

    The new pattern was obtained using qat.synthopline.

    Args:
        nb_fermionic_modes (int): Number of qubits.
        ncycles (int): Number of LDCA cycles.
        eigstate_ind (Optional[int]): Eigenstate index. Defaults to 0.
        slater (Optional[bool]): Whether to only include excitation-preserving rotations.
                                Defaults to False.

    Return:
    :class:`~qat.core.Circuit`
    """

    circ = make_ldca_circ(nb_fermionic_modes, ncycles, eigstate_ind=eigstate_ind, slater=slater)

    try:
        # pylint: disable=import-outside-toplevel
        from qat.pbo import GraphCircuit, VAR

    except ModuleNotFoundError:
        warnings.warn("The compressed LDCA circuit is available only for the QLM. Rolling back to standard LDCA circuit")
        return circ

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
