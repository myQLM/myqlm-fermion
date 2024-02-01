# -*- coding: utf-8 -*-
"""
Phase estimation functions
"""

import inspect
from typing import Optional, Tuple, Union
import numpy as np

import qat.comm.exceptions.ttypes as exceptions_types
from qat.lang.AQASM import Program, X, QRoutine, H, PH, QInt
from qat.lang.AQASM.qftarith import IQFT
from qat.qpus import get_default_qpu
from qat.core import Observable

from .hamiltonians import ElectronicStructureHamiltonian, SpinHamiltonian
from .util import construct_Rk_routine
from .transforms import (
    transform_to_jw_basis,
    transform_to_bk_basis,
    transform_to_parity_basis,
)


def perform_phase_estimation(
    H_el: ElectronicStructureHamiltonian,
    n_phase_bits: int,
    n_trotter_steps: int,
    init_vec: Optional[str] = None,
    n_adiab_steps: Optional[int] = 0,
    E_target: Optional[float] = 0,
    size_interval: Optional[float] = 2.0,
    basis_transform: Optional[str] = "jordan-wigner",
    qpu=None,
    n_shots: Optional[int] = 0,
    verbose: Optional[bool] = False,
) -> Tuple[float, float]:
    r"""
    Perform quantum phase estimation (QPE) on an :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian`. This Hamiltonian is
    transformed to the computational basis via a Jordan-Wigner transformation and approximated via first order trotterization.
    Other transformations like parity and Bravyi-Kitaev are also possible.

    When providing an initial state one can specify it either as a string composed of zeros and ones, or as a
    :class:`~qat.lang.AQASM.routines.QRoutine` which will produce it. The QPE is meant to start from an eigenstate of the
    Hamiltonian, however, knowing apriori even one eigenstate of the system may be challenging. Therefore, this function comes with
    adiabatic state preparation - an optional preliminary step to create an eigenstate of the Hamiltonian :math:`H`.
    This step consists in performing QPE :code:`n_adiab_steps` number of times, but not to read the phase bits (it is set to
    only one), but rather to collapse the system to an eigenstate (read from the data bits). The first of the series of QPE
    executions starts from the lowest energy eigenstate of the Hamiltonian composed of :math:`h_{pq}`. Then, :math:`h_{pq}` is
    linearly transformed to :math:`H` and at each new step we start from the eigenstate of the Hamiltonian of the previous step.
    This guarantees that when the main QPE routine starts, it will do so from an eigenstate of the full :math:`H`.

    Usually, energies lie outside the range :math:`(-\frac{2\pi}{t}, 0)`. However, this range can be adjusted by searching inside
    the window :math:`(E_{target} - \frac{\Delta}{2}, E_{target} + \frac{\Delta}{2})` with :math:`E_{target}` and :math:`\Delta`
    specified by :code:`E_target` and :code:`size_interval`, respectively. It is suggested to always start from a large size
    interval and unbiased target energy like :math:`0` thus enclosing many of the eigenenergies including the desired one. One can
    then narrow the window around an already found eigenenergy for a better precision. Working with a window not enclosing an
    eigenenergy would still evaluate to a result, but it may be misleading.

    .. warning::

        * Regarding the adiabatic state preparation, if the lowest energy eigenstate of the first-step Hamiltonian :math:`h_{pq}` is also an eigenstate of the whole :math:`H`, the system will remain in it until the end of the whole adiabatic stage. Hence, this eigenstate may not be the one of the lowest energy anymore.
        * As a rule of thumb, if small changes to the interval cause considerable deviations in the energy, that's a sign that the window is too small or a different target energy may be better.

    Args:
        H_el (ElectronicStructureHamiltonian): An electronic-structure Hamiltonian.
        n_phase_bits (int): Number of qubits for the phase evaluation. The larger it is, the
            more accurate is the result.
        n_trotter_steps (int): Number of first order trotterization steps. For good phase estimation it
            should also increase if n_phase_bits is increased.
        init_vec (Optional[str]): Initial vector specified in the computational basis as a
            string - '01101' for example. Starting from ``|0..0>`` an X will be applied to the respective
            qubits so as to produce the provided vector. This vector will enter the adiabatic state
            preparation routine if n_adiab_steps is not 0 or will be given straight to the main QPE routine.
        n_adiab_steps (Optional[int]): Number of steps to pass from the part of the Hamiltonian containing
            only c_p^dagger * c_p terms (which is diagonal and fast to deal with) to the Hamiltonian of interest.
        E_target (Optional[float]): Expected energy. If unknown, set to 0.
        size_interval (Optional[float]): Size :math:`\Delta` of the interval one thinks the value
            of the energy is in: :math:`E \in [E_\mathrm{target}-\Delta/2, E_\mathrm{target}+\Delta/2]`
            If no idea take :math:`\Delta =2 E_\mathrm{max}`, with :math:`E_\mathrm{max}` an upper
            bound of the energy.
        basis_transform (Optional[str]): Transformation to go from :class:`qat.fermion.hamiltonians.ElectronicStructureHamiltonian`
            into a :class:`qat.fermion.hamiltonians.SpinHamiltonian`: one can use the "jordan-wigner" (default),
            "bravyi-kitaev" or "parity" transformations.
        qpu (Optional[QPU]): QPU to use for computation. Will use by default the default installed QPU.

    Returns:
        Tuple[float, float]:
            - Energy found,
            - associated probability.

    Note:
        Usually, energies lie outside the range :math:`(-\frac{2\pi}{t}, 0)`. However, this range can be adjusted
        by specifying the arguments `E_target` and `size_interval` thus searching inside the window
        :math:`(E_{t} - \frac{\Delta}{2}, E_{target} + \frac{size\_interval}{2})`,
        where :math:`E_{t}` and :math:`\Delta` stand for. We suggest to always start from a large size interval
        and unbiased target energy like 0 thus enclosing many of the eigenenergies including the desired one.
        One can then narrow the window around an already found eigenenergy for a better precision.
        Experience shows that working with a window not enclosing an eigenenergy makes the QPE still output a result,
        but it is misleading.

    """

    if qpu is None:
        qpu = get_default_qpu()

    # A check for nqbits_adiab to be set if n_adiab_steps
    H_qbasis = None

    if basis_transform == "jordan-wigner":
        # _qbasis stands for qubit i.e. computational basis
        H_qbasis = transform_to_jw_basis(H_el)

    elif basis_transform == "bravyi-kitaev":
        # _qbasis stands for qubit i.e. computational basis
        H_qbasis = transform_to_bk_basis(H_el)

    elif basis_transform == "parity":
        # _qbasis stands for qubit i.e. computational basis
        H_qbasis = transform_to_parity_basis(H_el)

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

    # The constant energy already present in H_qbasis
    E_const = np.real(H_qbasis.constant_coeff)

    # We substract it, because it's not taken into account when we trotterize H
    E_target -= E_const
    size_interval = abs(size_interval)
    Emax = E_target + size_interval / 2
    H_evolution_time = 2 * np.pi / size_interval
    global_phase = Emax * H_evolution_time

    # "_hopping" is for the part of the H which has only hpp terms (no hpq or hpqrs)
    # extract the diagonal and return an empty array but with this diagonal.
    H_el_hopping_hpq = np.diag(np.diag(H_el.hpq))

    # "_f" stands for fermionic basis
    H_el_hopping_f = ElectronicStructureHamiltonian(H_el_hopping_hpq, hpqrs=None, constant_coeff=0.0)

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
                error_message = "Please specify the inital string vector with 0s and 1s only."

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
                message=("The state preparation acts on %s qubits " "but the Hamiltonian works with %s qubits.")
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

    # Adiabatic state preparation using intermediate QPE circuits
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
        for ind, (prob, state) in enumerate(reversed(sorted(list_states, key=lambda x: x[0]))):

            if ind < 5:
                print(state, state.value[0] / 2**n_phase_bits, prob)

    max_prob_state_int = np.argmax(probs)
    theta = max_prob_state_int / 2**n_phase_bits

    # exp(-i*H*t + i*Emax*t) |psi> = exp(2*pi*i*theta) |psi>
    energy = -2 * np.pi * theta / H_evolution_time + Emax
    energy += E_const

    return energy, np.max(probs)


def apply_adiabatic_state_prep(
    prog: Program,
    n_adiab_steps: int,
    H_el_hopping_qbasis,
    H_qbasis,
    n_trotter_steps: int,
    phase_reg,
    data_reg,
):
    nqbits_adiab = 1

    for t in np.linspace(0, 1, n_adiab_steps):

        H_current = (1 - t) * H_el_hopping_qbasis + t * H_qbasis

        pea_routine = build_qpe_routine_for_hamiltonian(H_current, nqbits_adiab, global_phase=0, n_trotter_steps=n_trotter_steps)

        # Use only the first nqbits_adiab of all the n_phase_bits
        prog.apply(pea_routine, phase_reg[:nqbits_adiab], data_reg)

        # Reset the qubits used for the adiabatic step to be ready for the actual QPE afterwards
        prog.reset(phase_reg[:nqbits_adiab])


def build_qpe_routine_for_hamiltonian(
    hamiltonian: Union[Observable, SpinHamiltonian],
    n_phase_bits: int,
    n_trotter_steps: Optional[int] = 1,
    global_phase: Optional[float] = 0,
    t: Optional[float] = 1,
) -> QRoutine:
    r"""
    Construct a phase estimation routine corresponding to a given spin Hamiltonian.

    Args:
        hamiltonian (Union[Observable, SpinHamiltonian]): Hamiltonian in the computational basis.
        n_phase_bits (int): The number of phase bits.
        n_trotter_steps (Optional[int]): The number of trotter steps. Defaults to 1.
        global_phase (Optional[float]): The global phase :math:`\phi` which the evolution
            operator :math:`U` starts from, i.e. :math:`U = e^{-iHt + i \phi}`.
            Default to 0.
        t (Optional[float]): The evolution time of the Hamiltonian. Default to 1.

    Returns:
        QRoutine: Quantum routine.

    """

    routine = QRoutine()
    phase_reg = routine.new_wires(n_phase_bits)
    data_reg = routine.new_wires(hamiltonian.nbqbits)

    # Hadamard wall
    for qb in range(n_phase_bits):
        routine.apply(H, phase_reg[qb])

    # Controlled unitaries along with a global phase application
    for j_ind in range(n_phase_bits):

        # Happens before the trotterization
        routine.apply(PH(global_phase * 2**j_ind), phase_reg[j_ind])
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

    # Now apply inverse QFT
    routine.apply(IQFT(n_phase_bits), phase_reg)

    return routine
