#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file qat/fermion/phase_estimation.py
@authors Thomas Ayral <thomas.ayral@atos.net>
         Grigori Matein <grigori.matein@atos.net>
@internal
@copyright 2021 Bull S.A.S. - All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief Phase estimation module
"""
import inspect
import numpy as np
from qat.core.variables import ArithExpression
import qat.comm.exceptions.ttypes as exceptions_types
from qat.fermion.hamiltonians import ElectronicStructureHamiltonian
from qat.fermion.transforms import (
    transform_to_jw_basis,
    transform_to_bk_basis,
    transform_to_parity_basis,
)
from qat.lang.AQASM import Program, X, QRoutine, H, PH, CNOT, RZ, RX, QInt
from qat.lang.AQASM.qftarith import IQFT
from qat.qpus import LinAlg


def perform_phase_estimation(
    H_el,
    n_phase_bits,
    n_trotter_steps,
    init_vec=None,
    n_adiab_steps=0,
    E_target=0,
    size_interval=2,
    basis_transform="jordan-wigner",
    qpu=LinAlg(),
    n_shots=0,
    verbose=False,
):
    r"""
    Perform quantum phase estimation (QPE) on an :class:`~qat.fermion.ElectronicStructureHamiltonian`. This Hamiltonian is transformed to the computational basis via a Jordan-Wigner transformation and approximated via first order trotterization. Other transformations like parity and Bravyi-Kitaev are also possible.

    When providing an initial state one can specify it either as a string composed of zeros and ones, or as a :class:`~qat.lang.AQASM.routines.QRoutine` which will produce it. The QPE is meant to start from an eigenstate of the Hamiltonian, however, knowing apriori even one eigenstate of the system may be challenging. Therefore, this function comes with adiabatic state preparation - an optional preliminary step to create an eigenstate of the Hamiltonian :math:`H`. This step consists in performing QPE :code:`n_adiab_steps` number of times, but not to read the phase bits (it is set to only one), but rather to collapse the system to an eigenstate (read from the data bits). The first of the series of QPE executions starts from the lowest energy eigenstate of the Hamiltonian composed of :math:`h_{pq}`. Then, :math:`h_{pq}` is linearly transformed to :math:`H` and at each new step we start from the eigenstate of the Hamiltonian of the previous step. This guarantees that when the main QPE routine starts, it will do so from an eigenstate of the full :math:`H`.

    Usually, energies lie outside the range :math:`(-\frac{2\pi}{t}, 0)`. However, this range can be adjusted by searching inside the window :math:`(E_{target} - \frac{\Delta}{2}, E_{target} + \frac{\Delta}{2})` with :math:`E_{target}` and :math:`\Delta` specified by :code:`E_target` and :code:`size_interval`, respectively. It is suggested to always start from a large size interval and unbiased target energy like :math:`0` thus enclosing many of the eigenenergies including the desired one. One can then narrow the window around an already found eigenenergy for a better precision. Working with a window not enclosing an eigenenergy would still evaluate to a result, but it may be misleading.

    .. warning::
        Regarding the adiabatic state preparation, if the lowest energy eigenstate of the first-step Hamiltonian :math:`h_{pq}` is also an eigenstate of the whole :math:`H`, the system will remain in it until the end of the whole adiabatic stage. Hence, this eigenstate may not be the one of the lowest energy anymore.

    .. warning::
        As a rule of thumb, if small changes to the interval cause considerable deviations in the energy, that's a sign that the window is too small or a different target energy may be better.


    Args:
        H_el (:class:`~qat.fermion.ElectronicStructureHamiltonian`): an electronic-structure Hamiltonian
        n_phase_bits (int): the number of qubits for the phase evaluation. The larger it is, the
            more accurate is the result.
        n_trotter_steps (int): number of first order trotterization steps. For good phase estimation it
            should also increase if n_phase_bits is increased.
        init_vec (string, optional): initial vector specified in the computational basis as a
            string - '01101' for example. Starting from |0..0> an X will be applied to the respective
            qubits so as to produce the provided vector. This vector will enter the adiabatic state
            preparation routine if n_adiab_steps is not 0 or will be given straight to the main QPE routine
        n_adiab_steps (int, optional): number of steps to pass from the part of the Hamiltonian containing
            only c_p^dagger * c_p terms (which is diagonal and fast to deal with) to the Hamiltonian of interest.
        E_target (float, optional): expected energy. If no idea, take 0
        size_interval (float, optional): the size :math:`\Delta` of the interval one thinks the value
            of the energy is in: :math:`E \in [E_\mathrm{target}-\Delta/2, E_\mathrm{target}+\Delta/2]`
            If no idea take :math:`\Delta =2 E_\mathrm{max}`, with :math:`E_\mathrm{max}` an upper
            bound of the energy.
        basis_transform (string, optional): transformation to go from :class:`qat.fermion.ElectronicStructureHamiltonian`
            into a :class:`qat.fermion.Hamiltonian`: one can use the "jordan-wigner" (default),
            "bravyi-kitaev" or "parity" transformations.
        qpu (QPU, optional): a QPU to use for computation, default is :class:`~qat.linalg.LinAlg`.

    Returns:
        float, float: energy found, assosiated probability

    """

    #         Usually, energies lie outside the range :math:`(-\frac{2\pi}{t}, 0)`. However, this range can be adjusted by specifying the arguments `E_target` and `size_interval` thus searching inside the window :math:`(E_{t} - \frac{\Delta}{2}, E_{target} + \frac{size_interval}{2})`, where :math:`E_{t}` and :math:`\Delta` stand for . We suggest to always start from a large size interval and unbiased target energy like 0 thus enclosing many of the eigenenergies including the desired one. One can then narrow the window around an already found eigenenergy for a better precision. Experience shows that working with a window not enclosing an eigenenergy makes the QPE still output a result, but it is misleading.

    # A check for nqbits_adiab to be set if n_adiab_steps
    H_qbasis = None
    if basis_transform == "jordan-wigner":
        H_qbasis = transform_to_jw_basis(
            H_el
        )  # _qbasis stands for qubit i.e. computational basis
    elif basis_transform == "bravyi-kitaev":
        H_qbasis = transform_to_bk_basis(
            H_el
        )  # _qbasis stands for qubit i.e. computational basis
    elif basis_transform == "parity":
        H_qbasis = transform_to_parity_basis(
            H_el
        )  # _qbasis stands for qubit i.e. computational basis
    else:
        current_line_no = inspect.stack()[0][2]
        raise exceptions_types.QPUException(
            code=exceptions_types.ErrorType.INVALID_ARGS,
            modulename="qat.fermion",
            message="Unrecognised transformation.",
            file=__file__,
            line=current_line_no,
        )
    n_qubits_H = H_qbasis.nbqbits

    E_const = np.real(
        H_qbasis.constant_coeff
    )  # the constant energy already present in H_qbasis
    E_target -= E_const  # we substract it, because it's not taken into account when we trotterize H
    size_interval = abs(size_interval)
    Emax = E_target + size_interval / 2
    H_evolution_time = 2 * np.pi / size_interval
    global_phase = Emax * H_evolution_time

    # "_hopping" is for the part of the H which has only hpp terms (no hpq or hpqrs)
    H_el_hopping_hpq = np.diag(
        np.diag(H_el.hpq)
    )  # extract the diagonal and return an empty array but with this diagonal.
    H_el_hopping_f = ElectronicStructureHamiltonian(
        H_el_hopping_hpq, hpqrs=None, constant_coeff=0.0
    )  # "_f" stands for fermionic basis
    H_el_hopping_qbasis = transform_to_jw_basis(H_el_hopping_f)

    # Initialize the program
    prog = Program()
    phase_reg = prog.qalloc(n_phase_bits, class_type=QInt, reverse_bit_order=False)
    data_reg = prog.qalloc(n_qubits_H)

    # Prepare the inital state of the total circuit if none is provided
    if init_vec is None:

        # Extract and sort all the N orbital energies of H_el_hopping_f
        orbital_energies = np.diag(H_el_hopping_hpq)

        # The corresponding eigenvec can be constructed by taking an empty array of size n_qubits
        # and putting '1' at the positions of these negative orbital energies
        lowest_E_eigvec = (np.array(orbital_energies) <= 0).astype(int)
        init_vec = lowest_E_eigvec

    # Check that the inital vector is properly specified - as a string of 0s and 1s
    elif isinstance(init_vec, str):
        error_message = None
        if not isinstance(init_vec, str):
            error_message = (
                "If an initial vector is given, it should be specified as a "
                + "string of the form '01101', i.e. in the computational basis."
            )
        elif len(init_vec) != n_qubits_H:
            error_message = "The length of the vector does not match the Hamiltonian."
        else:
            if any(c not in "01" for c in init_vec):
                error_message = (
                    "Please specify the inital string vector with 0s and 1s only."
                )
        if error_message:
            current_line_no = inspect.stack()[0][2]
            raise exceptions_types.QPUException(
                code=exceptions_types.ErrorType.INVALID_ARGS,
                modulename="qat.fermion",
                message=error_message,
                file=__file__,
                line=current_line_no,
            )
    elif isinstance(init_vec, QRoutine):
        if init_vec.arity != n_qubits_H:
            current_line_no = inspect.stack()[0][2]
            raise exceptions_types.QPUException(
                code=exceptions_types.ErrorType.INVALID_ARGS,
                modulename="qat.fermion",
                message=(
                    "The state preparation acts on %s qubits "
                    "but the Hamiltonian works with %s qubits."
                )
                % (init_vec.arity, n_qubits_H),
                file=__file__,
                line=current_line_no,
            )
        prog.apply(init_vec, data_reg)

    # If a string, produce the initial state in the circuit
    if isinstance(init_vec, str):
        # Apply an X gate to every qubit needing to be in state '1'.
        for i, state in enumerate(init_vec):
            if state == "1":
                prog.apply(X, data_reg[i])
    #         print(init_vec)

    # Adiabatic state preparation using intermediate QPE circuits - in an outer function
    # so as to be separately tested
    apply_adiabatic_state_prep(
        prog,
        n_adiab_steps,
        H_el_hopping_qbasis,
        H_qbasis,
        n_trotter_steps,
        phase_reg,
        data_reg,
    )

    # Actual phase estimation with the QPE routine
    pea_routine = build_qpe_routine_for_hamiltonian(
        H_qbasis,
        n_phase_bits,
        global_phase=global_phase,
        t=H_evolution_time,
        n_trotter_steps=n_trotter_steps,
    )
    prog.apply(pea_routine, phase_reg, data_reg)

    # Generate the corresponding circuit and execute it
    circ = prog.to_circ()
    res = qpu.submit(circ.to_job(qubits=phase_reg, nbshots=n_shots))

    # Store the output probabilities in a vector
    probs = np.zeros(2**n_phase_bits)
    list_states = []
    for sample in res:
        list_states.append((sample.probability, sample.state))
        probs[sample.state.value[0]] = sample.probability

    if verbose:
        # Print the first 5 states (sorted by decreasing probabilities)
        for ind, (prob, state) in enumerate(
            reversed(sorted(list_states, key=lambda x: x[0]))
        ):

            if ind < 5:
                print(state, state.value[0] / 2**n_phase_bits, prob)

    max_prob_state_int = np.argmax(probs)
    theta = max_prob_state_int / 2**n_phase_bits
    energy = (
        -2 * np.pi * theta / H_evolution_time + Emax
    )  # exp(-i*H*t + i*Emax*t) |psi> = exp(2*pi*i*theta) |psi>
    energy += E_const
    return energy, np.max(probs)


def apply_adiabatic_state_prep(
    prog,
    n_adiab_steps,
    H_el_hopping_qbasis,
    H_qbasis,
    n_trotter_steps,
    phase_reg,
    data_reg,
):
    nqbits_adiab = 1
    for t in np.linspace(0, 1, n_adiab_steps):
        H_current = (1 - t) * H_el_hopping_qbasis + t * H_qbasis

        pea_routine = build_qpe_routine_for_hamiltonian(
            H_current, nqbits_adiab, global_phase=0, n_trotter_steps=n_trotter_steps
        )
        prog.apply(
            pea_routine, phase_reg[:nqbits_adiab], data_reg
        )  # use only the first nqbits_adiab of all the n_phase_bits

        # Reset the qubits used for the adiabatic step to be ready for
        # the actual QPE afterwards
        prog.reset(phase_reg[:nqbits_adiab])


def build_qpe_routine_for_hamiltonian(
    hamiltonian, n_phase_bits, n_trotter_steps=1, global_phase=0, t=1
):
    """
    Construct a phase estimation routine corresponding to a given spin Hamiltonian.

    Args:
        hamiltonian (Observable): a Hamiltonian in the computational basis
        n_phase_bits (int): the number of phase bits
        n_trotter_steps (int, optional): the number of trotter steps. Defaults to 1.
        global_phase (float, optional): the global phase :math:`\phi` which the evolution
            operator :math:`U` starts from, i.e. :math:`U = e^{-iHt + i \phi}`.
            Defaults to 0.
        t (float, optional): the evolution time of the Hamiltonian. Default is 1.

    Returns:
        QRoutine: a quantum routine
    """
    routine = QRoutine()
    phase_reg = routine.new_wires(n_phase_bits)
    data_reg = routine.new_wires(hamiltonian.nbqbits)

    # Hadamard wall
    for qb in range(n_phase_bits):
        routine.apply(H, phase_reg[qb])

    # Controlled unitaries along with a global phase application
    for j_ind in range(n_phase_bits):
        routine.apply(
            PH(global_phase * 2**j_ind), phase_reg[j_ind]
        )  # happens before the trotterization
        for _ in range(n_trotter_steps):
            for term in hamiltonian.terms:
                if np.imag(term.coeff) > 1e-10:
                    raise Exception(
                        "There was a non-real term in the Hamiltonian translated to the"
                        " qubit basis. All the terms should be real, coming from a"
                        " hermitian H."
                    )
                theta = np.real(term.coeff) * 2 ** (j_ind + 1) * t / n_trotter_steps
                Rk_routine = construct_Rk_routine(term.op, term.qbits, theta)
                routine.apply(
                    Rk_routine.ctrl(),
                    phase_reg[j_ind],
                    [data_reg[qb] for qb in term.qbits],
                )

    # now apply inverse QFT
    routine.apply(IQFT(n_phase_bits), phase_reg)

    return routine


def construct_Rk_routine(ops, qbits, theta):
    r"""Implement

    .. math::
         R_k(\theta) = \exp\left(-i \frac{\theta}{2} P_k\right)

    with P_k a Pauli string

    Args:
        ops (str): Pauli operators (e.g X, Y, ZZ, etc.)
        qbits (list<int>): qubits on which they act
        theta (Variable): the abstract variable

    Returns:
        QRoutine

    Notes:
        the indices of the wires of the QRoutine are relative
        to the smallest index in qbits (i.e always start at qb=0)
    """
    if not isinstance(theta, ArithExpression):
        theta = np.real(theta)

    qrout = QRoutine()
    qbits = qrout.new_wires(len(qbits))
    with qrout.compute():
        for op, qbit in zip(ops, qbits):
            if op == "X":
                qrout.apply(H, qbit)
            if op == "Y":
                qrout.apply(RX(np.pi / 2), qbit)
        for ind_qb in range(len(qbits) - 1):
            qrout.apply(CNOT, qbits[ind_qb], qbits[ind_qb + 1])
    qrout.apply(RZ(theta), qbits[-1])
    qrout.uncompute()  # uncompute() applies U^dagger,
    # with U the unitary corresponding to the gates applied within the "with XX.compute()" context

    return qrout