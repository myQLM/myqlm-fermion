# -*- coding: utf-8 -*-
"""
Quantum Subspace Expansion (QSE)
"""

import itertools
from typing import List, Tuple, Optional
import numpy as np
from scipy.linalg import eig

from qat.core import Term, Circuit, Observable
from qat.core.qpu import QPUHandler

from ..hamiltonians import SpinHamiltonian, FermionHamiltonian


def apply_quantum_subspace_expansion(
    hamiltonian: SpinHamiltonian,
    state_prep_circ: Circuit,
    expansion_operators: List[Observable],
    qpu: QPUHandler,
    nbshots: int = 0,
    threshold: float = 1e-15,
    return_matrices: bool = False,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    r"""Apply quantum subspace expansion (QSE) to the given Hamiltonian.

    QSE is a method that improves the quality of noisy results at
    the cost of additional measurements that help write the Hamiltonian
    in the small subspace where it can be diagonalized classically.

    If :math:`\langle \Psi^\star | \hat{H} | \Psi^\star \rangle` is the
    VQE result, the projected Hamiltonian matrix is built from

    .. math::

        H^{(s)}_{i,j} =   \langle \Psi^\star |
                    \hat{O}_i^\dagger \hat{H} \hat{O}_j
                    | \Psi^\star \rangle

    where :math:`\hat{O}_i` is an element of `expansion_operators`.

    Then the generalized eigenvalue problem

    .. math::

        H^{(s)} \vec{x} = E S \vec{x}

    is solved, with :math:`S` the overlap matrix:

    .. math::

        S_{i,j} = \langle \Psi^\star | \hat{O}_i^\dagger \hat{O}_j | \Psi^\star \rangle

    Args:

        hamiltonian (SpinHamiltonian): The Hamiltonian in its spin representation.
        state_prep_circ (Circuit): The state preparation circuit.
        expansion_operators (list<SpinHamiltonian>): The set of operators :math:`{O_i}_i`
            generating the subspace of interest.
        qpu (QPUHandler): The QPU.
        nbshots (int, optional): The number of shots. Defaults to 0:
            infinite number of shots.
        threshold (float, optional): The numerical threshold.
        return_matrices (bool, optional): If set to :code:`True`, the
            function returns the matrices :math:`H^{(s)}` and :math:`S`. Defaults to False.

    Returns:

        e_qse (float):  Improved energy provided by the QSE procedure.
        matrix_h (Optional[np.ndarray]): The subspace Hamiltonian :math:`H^{(s)}`. Only if :code:`return_matrices` is True.
        matrix_s (Optional[np.ndarray]): The overlap matrix :math:`S`.  Only if :code:`return_matrices` is True.

    Example:

    .. run-block:: python

        import numpy as np
        from qat.core import Term
        from qat.fermion import SpinHamiltonian
        from qat.lang.AQASM import Program, RY, CNOT, RZ
        from qat.qpus import get_default_qpu
        from qat.plugins import SeqOptim

        # We instantiate the Hamiltonian we want to approximate the ground state energy of
        hamiltonian = SpinHamiltonian(2, [Term(1, op, [0, 1]) for op in ["XX", "YY", "ZZ"]])

        # We construct the variational circuit (ansatz)
        prog = Program()
        reg = prog.qalloc(2)
        theta = [prog.new_var(float, '\\theta_%s'%i) for i in range(3)]
        RY(theta[0])(reg[0])
        RY(theta[1])(reg[1])
        RZ(theta[2])(reg[1])
        CNOT(reg[0], reg[1])
        circ = prog.to_circ()

        # Construct a (variational) job with the variational circuit and the observable
        job = circ.to_job(observable=hamiltonian,
                        nbshots=0)

        # We now build a stack that can handle variational jobs
        qpu = get_default_qpu()

        optimizer = SeqOptim(ncycles=10, x0=[0, 0.5, 0])
        stack = optimizer | qpu

        # We submit the job and print the optimized variational energy (the exact GS energy is -3)
        result = stack.submit(job)
        E_min = -3
        print("E(VQE) = %s (err = %s %%)"%(result.value, 100*abs((result.value-E_min)/E_min)))
        e_vqe = result.value

        # We use the optimal parameters found by VQE
        opt_circ = circ.bind_variables(eval(result.meta_data["parameter_map"]))

        expansion_operators = [SpinHamiltonian(2, [], 1.0),
                            SpinHamiltonian(2, [Term(1., "ZZ", [0, 1])])]

        from qat.fermion.chemistry.qse import apply_quantum_subspace_expansion
        e_qse = apply_quantum_subspace_expansion(hamiltonian,
                                                opt_circ,
                                                expansion_operators,
                                                qpu,
                                                return_matrices=False)

        print("E(QSE) = %s (err = %s %%)"%(e_qse, 100*abs((e_qse-E_min)/E_min)))

    """

    matrix_h, matrix_s = _build_quantum_subspace_expansion(
        hamiltonian,
        state_prep_circ,
        expansion_operators,
        qpu,
        nbshots=nbshots,
        threshold=threshold,
    )
    eig_vals = eig(matrix_h, matrix_s)
    e_qse = min([x.real for x in eig_vals[0]])

    if return_matrices:
        return e_qse, matrix_h, matrix_s
    return e_qse


def _build_quantum_subspace_expansion(
    hamiltonian: SpinHamiltonian,
    state_prep_circ: Circuit,
    expansion_operators: List[Observable],
    qpu: QPUHandler,
    nbshots: int = 0,
    threshold: float = 1e-15,
) -> Tuple[np.matrix, np.matrix]:
    r"""Build the quantum subspace expansion (QSE) of the Hamiltonian as
    a matrix and the corresponding overlap matrix.

    QSE is a method that improve the quality of noisy results by
    mapping the Hamiltonian to a subspace and diagonalizing it
    classically.

    If :math:`\langle \Psi^\star | \hat{H} | \Psi^\star \rangle` is the
    VQE result, the projected Hamiltonian matrix :math:`H` and the
    overlap matrix :math:`S` are built from
    .. math::

        H_{i,j} &=   \langle \Psi^\star |
                    \hat{sigma}_i^\dagger \hat{H} \hat{sigma}_j
                    | \Psi^\star \rangle \\

        S_{i,j} &=   \langle \Psi^\star |
                    \hat{sigma}_i^\dagger \hat{sigma}_j
                    | \Psi^\star \rangle

    where :math:`\hat{\sigma}` is an element of ``expansion_operators``.

    Args:

        hamiltonian (SpinHamiltonian): The Hamiltonian.
        state_prep_circ (Circuit): The state prep circuit.
        qpu (QPUHandler): The qpu.
        expansion_operators (List[Observable]): The list of operators.
            generating the subspace of interest.
        nbshots (int, optional): The number of shots.
        threshold (float, optional): The numerical threshold.

    Returns:
        matrix_h (np.matrix): The subspace Hamiltonian.
        matrix_s (np.matrix): The overlap matrix of the subspace.
    """
    dim = len(expansion_operators)

    matrix_h = np.asmatrix(np.zeros((dim, dim), dtype=np.float64))
    matrix_s = np.asmatrix(np.zeros((dim, dim), dtype=np.float64))

    # Selection of the left operator to be applied:
    for (i, op_l) in enumerate(expansion_operators):

        # Selection of the right operator to be applied:
        for (j, op_r) in enumerate(expansion_operators):

            # Construction of the expanded Hamiltonian and of the overlap
            # matrix:
            h_expanded = op_l.dag() * hamiltonian * op_r
            overlap = op_l.dag() * op_r

            # Measurement of the expectation value:
            scalar_h = qpu.submit(state_prep_circ.to_job(observable=h_expanded, nbshots=nbshots)).value
            scalar_s = qpu.submit(state_prep_circ.to_job(observable=overlap, nbshots=nbshots)).value

            matrix_h[i, j] = scalar_h.real if abs(scalar_h) > threshold else 0
            matrix_s[i, j] = scalar_s.real if abs(scalar_s) > threshold else 0

    return matrix_h, matrix_s


def build_linear_pauli_expansion(pauli_gates: List[str], nb_qubits: int) -> List[Observable]:
    r"""Builds first-order all-qubit expansion from the listed Pauli
    gates.

    For example, with ``pauli_gates = ['X', 'Y'], nb_qubits = 2``, the
    function outputs a list with ``QubitOperators X0, X1, Y1, Y0``.

    Args:
        pauli_gates (List[str]): The Pauli gates to use in the expansion,
            as a list of strings among ``'I' 'X', 'Y', 'Z'``.
        nb_qubits (int): The number of qubits.

    Returns:
        expansion_operators (List[Observable]): The expansion as a
            list of QubitOperator acting on ``nb_qubits`` qubits.
    """
    expansion_operators = []

    for (pauli, qbit) in itertools.product(pauli_gates, range(nb_qubits)):
        op = SpinHamiltonian(nb_qubits, [Term(1.0, pauli, [qbit])])
        expansion_operators.append(op)

    return expansion_operators
