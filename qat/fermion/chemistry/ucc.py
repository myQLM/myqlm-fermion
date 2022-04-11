#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools imported from notebook

"""
import itertools
import numpy as np
from bitstring import BitArray
from typing import Any, List, Tuple, Callable, Optional, Dict

from qat.core import Term
from qat.lang.AQASM import X, Program
from ..trotterisation import make_spin_hamiltonian_trotter_slice
from ..hamiltonians import Hamiltonian, ElectronicStructureHamiltonian
from ..util import tobin
from ..trotterisation import make_trotterisation_routine


def transform_integrals_to_new_basis(
    one_body_integrals: np.ndarray, two_body_integrals: np.ndarray, U_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Change one and two body integrals (indices p, q...) to
    new basis (indices i, j...) using transformation U such that

    .. math::
        \hat{c}_{i}=\sum_{q}U_{qi}c_{q}

    i.e

    .. math::

        \hat{I}_{ij} =\sum_{pq}U_{pi}I_{pq}U_{jq}^{\dagger}
        \hat{I}_{ijkl}=\sum_{pqrs}U_{pi}U_{qj}I_{pqrs}U_{kr}^{\dagger}U_{ls}^{\dagger}

    Args:
        one_body_integrals (np.ndarray): one-body integrals :math:`I_{pq}`.
        two_body_integrals (np.ndarray): two-body integrals :math:`I_{pqrs}`.
        U_mat (np.ndarray): transformation matrix :math:`U`.

    Returns:
        h_hat_ij (np.ndarray): one-body integrals :math:`\hat{I}_{ij}`.
        h_hat_ijkl (np.ndarray): two-body integrals :math:`\hat{I}_{ijkl}`.
    """

    U_matd = np.conj(U_mat.T)

    h_hat_ij = np.einsum("pi,pq,jq", U_mat, one_body_integrals, U_matd)
    h_hat_ijkl = np.einsum(
        "pi,qj,pqrs,kr,ls", U_mat, U_mat, two_body_integrals, U_matd, U_matd
    )

    return h_hat_ij, h_hat_ijkl


def compute_core_constant(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    occupied_indices: List[int],
) -> float:
    """Compute the core constant from the one- and two-body integrals.

    Args:
        one_body_integrals (np.ndarray): One-body integrals.
        two_body_integrals (np.ndarray): Two-body integrals.
        occupied_indices (List[int]): Occupied indices.

    Returns:
        float: Core constant.
    """

    core_constant = 0.0
    for i in occupied_indices:
        core_constant += 2 * one_body_integrals[i, i]
        for j in occupied_indices:
            core_constant += (
                2 * two_body_integrals[i, j, j, i] - two_body_integrals[i, j, i, j]
            )

    return core_constant


def compute_active_space_integrals(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    active_indices: List[int],
    occupied_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""Restrict one- and two-body integrals for given list of active indices.

    .. math::

        \forall u,v\in \mathcal{A},\; I^{(a)}_{uv} = I_{uv} + \sum_{i\in \mathcal{O}} 2 I_{i,u,v,i} - I_{i,u,i,v}

        \forall u,v,w,x \in \mathcal{A}, I^{(a)}_{uvwx} = I_{uvwx}

        c^{(a)} = c + \sum_{i\in\mathcal{O}) I_{ii} + \sum_{ij\in\mathcal{O} 2I_{ijji} - I_{ijij}

    Args:
        one_body_integrals (np.ndarray): 2D array of one-body integrals :math:`I_{uv}`.
        two_body_integrals (np.ndarray): 4D array of two-body integrals :math:`I_{uvwx}`.
        active_indices (List[int]): Active indices.
        occupied_indices (List[int]): Occupied indices.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - 2D array of one-body integrals :math:`I_{uv}^{(a)}`.
            - 4D array of two-body integrals :math:`I_{uvwx}^{(a)}`.
            - core constant :math:`c^{(a)}`.
    """
    # Modified core constant
    core_constant = compute_core_constant(
        one_body_integrals, two_body_integrals, occupied_indices
    )

    # Modified one electron integrals
    one_body_integrals_new = np.copy(one_body_integrals)
    for u, v, i in itertools.product(active_indices, active_indices, occupied_indices):
        one_body_integrals_new[u, v] += (
            2 * two_body_integrals[i, u, v, i] - two_body_integrals[i, u, i, v]
        )

    # Restrict integral ranges
    return (
        core_constant,
        one_body_integrals_new[np.ix_(active_indices, active_indices)],
        two_body_integrals[
            np.ix_(active_indices, active_indices, active_indices, active_indices)
        ],
    )


def _one_body_integrals_to_h(one_body_integrals: np.ndarray) -> np.ndarray:
    """Converts one-body integrals to one-body (spin-resolved) coefficient.

    Args:
        one_body_integrals (np.ndarray): One body integrals.

    Returns:
        np.ndarray: One-body coefficient.
    """

    nb_qubits = 2 * one_body_integrals.shape[0]

    one_body_coefficients = np.zeros((nb_qubits, nb_qubits), dtype=np.complex128)

    # Build the coefficients of the Hamiltonian:
    for p, q in itertools.product(range(nb_qubits // 2), repeat=2):
        y = one_body_integrals[p, q]

        # Populate 1-body coefficients. Require p and q have same spin.
        for sp in [0, 1]:
            one_body_coefficients[2 * p + sp, 2 * q + sp] = y

    return one_body_coefficients


def _two_body_integrals_to_h(two_body_integrals: np.ndarray) -> np.ndarray:
    """Converts two-body integrals to two-body (spin-resolved) coefficient.

    Args:
        two_body_integrals (np.ndarray): Two body integrals.

    Returns:
        np.ndarray: Two-body coefficient.
    """

    nb_qubits = 2 * two_body_integrals.shape[0]

    two_body_coefficients = np.zeros(
        (nb_qubits, nb_qubits, nb_qubits, nb_qubits), dtype=np.complex128
    )

    # Build the coefficients of the Hamiltonian:
    for p, q in itertools.product(range(nb_qubits // 2), repeat=2):

        # Continue looping to prepare 2-body coefficients.
        for r, s in itertools.product(range(nb_qubits // 2), repeat=2):
            x = two_body_integrals[p, q, r, s]

            # Require p,s and q,r to have same spin.

            # Handle mixed spins.
            for sp in [0, 1]:
                two_body_coefficients[
                    2 * p + sp, 2 * q + (1 - sp), 2 * r + (1 - sp), 2 * s + sp
                ] = x

            # Handle same spins.
            if p != q and r != s:
                for sp in [0, 1]:
                    two_body_coefficients[
                        2 * p + sp, 2 * q + sp, 2 * r + sp, 2 * s + sp
                    ] = x

    return two_body_coefficients


def convert_to_h_integrals(
    one_body_integrals: np.ndarray, two_body_integrals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert from :math:`I_{uv},I_{uvwx}` to :math:`h_{pq},h_{pqrs}`, with

    .. math::

        h_{u\sigma, v\sigma'} = I_{u, v} \delta_{\sigma, \sigma'}

        h_{u\sigma_1, v\sigma_2, w\sigma_2', x\sigma_1'} =  I_{uvwx} \left((1-\delta_{\sigma,\sigma'}) + \delta_{\sigma,\sigma'} (1-\delta_{u,v})(1-\delta_{w,x})   \right)

    and where the one- and two-body integrals are defined as:

    .. math::

        I_{uv}\equiv(u|h|v)=\int\mathrm{d}r\phi_{u}^{*}(r)T\phi_{v}(r)

    .. math::

        I_{uvwx}\equiv(ux|vw)=\iint\mathrm{d}r_{1}\mathrm{d}r_{2}\phi_{u}^{*}(r_{1})\phi_{x}(r_{1})v(r_{12})\phi_{v}^{*}(r_{2})\phi_{w}(r_{2})

    with :math:`T` (resp. :math:`v`) the one- (resp. two-) body potentials,
    and :math:`\phi_u(r)` is the molecular orbital wavefunction.

    The :math:`h` integrals are used to construct hamiltonians of the
    ElectronicStructureHamiltonian type

    Args:
        one_body_integrals (np.ndarray): 2D array of one-body integrals :math:`I_{uv}`.
        two_body_integrals (np.ndarray): 4D array of two-body integrals :math:`I_{uvwx}`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - the :math:`h_{pq}` integrals.
            - the :math:`h_{pqrs}` integrals.
    """

    one_body_coefficients = _one_body_integrals_to_h(one_body_integrals)
    two_body_coefficients = _two_body_integrals_to_h(two_body_integrals)

    return one_body_coefficients, two_body_coefficients


def build_cluster_operator(l_ex_op: List[Tuple[int]], nqbits: int) -> List[Hamiltonian]:
    r"""Builds the cluster operator and reduces the trial
    parametrization to match the selected excitation operators.

    The UCCSD cluster operator :math:`T` is defined (in normal-ordered form) as:

    .. math::

        T(\theta) = \sum_{a, i} \theta_a^i (a^\dagger_a a_i -
        a^\dagger_i a_a) + \sum_{a > b, i > j} \theta_{a, b}^{i, j}
        (a^\dagger_a a^\dagger_b a_i a_j - a^\dagger_i a^\dagger_j a_a
        a_b)

    where :math:`i, j` (resp. :math:`a, b`) indices occupied (resp.
    inoccupied) spin-orbitals. It is antihermitian.

    The function returns :math:`iT`.

    Finally, the trial MP2 parametrization is reduced to only store the
    non-zero parameters corresponding to the selected excitation
    parameters.

    Args:
        l_ex_op (List[Tuple[int]]): The list of of (a, b, i, j) and (a,
            i) tuples describing the excitation operators (without
            Hermitan conjugate, i.e. only excitation from unoccupied to
            occupied orbitals) to consider among the set associated to
            the active orbitals.
        nqbits (int): The total number of qubits.

    Returns:
        t_opti (List[Hamiltonian]): The cluster operator (times i) "iT"
            as a dictionary corresponding to each group of fermionic
            excitation operators parametrized identically.
    """
    t_opti = []

    for op_index in l_ex_op:
        current_excitation_op = []

        op_description, indices, indices_conj = None, None, None
        if len(op_index) == 2:
            op_description = "Cc"
            indices, indices_conj = list(op_index), [op_index[1], op_index[0]]

        else:  # i.e. len(op_index) == 4
            op_description = "CCcc"
            indices, indices_conj = list(op_index), [
                op_index[2],
                op_index[3],
                op_index[0],
                op_index[1],
            ]

        current_excitation_op.append(Term(1j, op_description, indices))
        current_excitation_op.append(Term(-1j, op_description, indices_conj))
        t_opti.append(Hamiltonian(nqbits=nqbits, terms=current_excitation_op))

    return t_opti


def construct_ucc_ansatz(
    cluster_ops: List[Hamiltonian], ket_hf: int, n_steps: int = 1
) -> Program:
    r"""Builds the parametric state preparation circuit implementing the
    provided cluster operator.

    The returned function maps :math:`\vec{\theta}` to a QRoutine
    describing :math:`Q` such as:

    .. math::

        Q \vert \vec{0} \rangle
            &= \vert \mathrm{UCC} (\vec{\theta}) \rangle \\
            &= e^{T(\vec{\theta})} \vert \mathrm{HF}\rangle

    Args:
        cluster_ops (List[Hamiltonian]): the cluster operators iT (note the i factor).
        ket_hf (int): The Hartree-Fock state in integer representation.
        n_steps(int): number of trotter steps.

    Returns:
        Program: The parametric program implementing the UCCSD method.
    """

    # Define system
    nqbits = cluster_ops[0].nbqbits
    n_ops = len(cluster_ops)

    # Get Hartree-Fock state binary representation
    ket_hf_init_sp = [int(c) for c in tobin(ket_hf, nqbits)]

    # Initialize a program
    prog = Program()
    reg = prog.qalloc(nqbits)

    # Initialize the HF state
    for j in range(nqbits):
        if int(ket_hf_init_sp[j]) == 1:
            prog.apply(X, reg[j])

    # Define the parameters to optimize
    theta = [prog.new_var(float, "\\theta_{%s}" % i) for i in range(n_ops)]

    # Define the Hamiltonian
    hamiltonian = sum([th * T for th, T in zip(theta, cluster_ops)])

    # Trotterize the Hamiltonian
    qrout = make_trotterisation_routine(
        hamiltonian, n_trotter_steps=n_steps, final_time=1
    )

    # Apply QRoutine onto the Program
    prog.apply(qrout, reg)

    return prog


def select_active_orbitals(
    noons: List[float],
    n_electrons: int,
    threshold_1: Optional[float] = 2e-2,
    threshold_2: Optional[float] = 2e-3,
) -> Tuple[List[int], List[int]]:
    r"""Selects the right active space and freezes core electrons
    according to their NOONs.

    This function is an implementation of the *Complete Active Space*
    (CAS) approach. It divides orbital space into sets of *active* and
    *inactive* orbitals, the occupation number of the latter remaining
    unchanged during the computation.


    Args:
        noons (np.ndarray): The natural orbital occupation numbers
            in descending order (from high occupations to low occupations)
        nb_e (int): The number of electrons.
        threshold_1 (Optional[float]): The upper threshold :math:`\varepsilon_1` on
            the NOON of an active orbital. Defaults to 0.02.
        threshold_2 (Optional[float]): The lower threshold :math:`\varepsilon_2` on
            the NOON of an active orbital. Defaults to 0.001.

    Returns:
        active_so (List[int]): The list of active spatial orbitals.
        inactive_occupied_so (List[int]): The list of core spatial orbitals.
    """

    # Initialize active and core space orbitals lists
    active_so, inactive_occupied_so = [], []

    for idx, noon in enumerate(noons):

        if noon >= 2 - threshold_1:

            if 2 * (idx + 1) < n_electrons:
                inactive_occupied_so.append(idx)

            else:
                active_so.append(idx)

        elif noon >= threshold_2:
            active_so.append(idx)

        else:
            break

    return active_so, inactive_occupied_so


def _theta_ab_ij(
    active_occupied_orbitals: List[int],
    active_unoccupied_orbitals: List[int],
    int2e: np.ndarray,
    orbital_energies: List[float],
    threshold: float = 1e-15,
) -> Dict[Tuple[int], float]:
    r"""Build the trial parametrization based upon a variational
    method known as second ordre Møller-Plesset (MP2).

    The Restricted Hartree-Fock (RHF) procedure provides the initial
    state and thus, its related parametrization would be a null vector.

    To improved upon it, a cost-efficient method is provided by the MP2
    method. Indeed, the latter is a post-Hartree-Fock method whose
    results are thus better than HF, and by identification, it is
    possible to find the MP2 values of the UCC parameters. They are
    given by the following equation:
    .. math::

        \theta_a^i = 0 \qquad
        \theta_{a, b}^{i, j} = \frac{h_{a, b, i, j} -
        h_{a, b, j, i}}{\epsilon_i + \epsilon_j -\epsilon_a -
        \epsilon_b}

    where :math:`i, j` (resp. :math:`a, b`) indices occupied (resp.
    inoccupied) spin-orbitals; math:`h_{p, q, r, s}` are the 2-electron
    molecular orbital integrals; math:`\epsilon_i` are the orbital
    energies.

    Note:
        The trial parametrization is stored as a dictionary: ``theta[(a,
        b, i, j)]`` describes the parameter :math:`\theta_{a, b}^{i,
        j}`of :math:`a^\dagger_a a^\dagger_b a_i a_j`.

    Args:
        active_occupied_orbitals (List[int]): The list of the active
            occupied orbitals.
        active_unoccupied_orbitals (List[int]): The list of the active
            unoccupied orbitals.
        int2e (np.ndarray): The 2-electron integrals corrected for
            and reduced to the active space.
        orbital_energies (List[float]): The vector of orbital
            energies.
        threshold (float): The numerical threshold used to
            nullify smaller terms through out the execution of the code.

    Returns:
        theta (Dict[Tuple[int], float]): The trial MP2 parametrization as
            a dictionary corresponding to the factors of each excitation
            operator (only the terms above ``threshold`` are stored).
    """
    theta = {}

    for (i, j, a, b) in itertools.product(
        active_occupied_orbitals,
        active_occupied_orbitals,
        active_unoccupied_orbitals,
        active_unoccupied_orbitals,
    ):

        if i != j and a != b:
            val_calc = (int2e[a, b, i, j] - int2e[a, b, j, i]) / (
                orbital_energies[i]
                + orbital_energies[j]
                - orbital_energies[a]
                - orbital_energies[b]
            )

            if abs(val_calc) >= threshold and abs(val_calc) != np.inf:

                if abs(val_calc.imag) < threshold:
                    theta[(a, b, i, j)] = val_calc.real

                else:
                    theta[(a, b, i, j)] = val_calc

    return theta


def _init_uccsd(
    nb_o: int,
    nb_e: int,
    int2e: np.ndarray,
    l_ao: List[int],
    orbital_energies: List[float],
) -> Tuple[np.ndarray, List[int], List[int], Dict[Tuple[int], float]]:
    r"""Executes the different (classical) methods whose results are
    needed to set up the state preparation and the Hamiltonian.

    Applying the Open-Shell Restricted Hartree-Fock (OS-RHF) procedure,
    the initial state is prepared by filling spin-orbitals as far as
    possible i.e. until there is no longer electron to place.

    The list of active occupied and unoccupied spin-orbitals are
    extracted from the (renumbered) active space. An orbital is occupied
    if and only if it is so after the OS-RHF procedure.

    The trial parametrization is efficiently improved upon the
    Hartree-Fock solution (which would set every initial parameter to
    zero) thanks to the following formula identifying the UCC parameters
    in the Møller-Plesset (MP2) solution :

    .. math::

        \theta_a^i = 0 \qquad
        \theta_{a, b}^{i, j} = \frac{h_{a, b, i, j} -
        h_{a, b, j, i}}{\epsilon_i + \epsilon_j -\epsilon_a -
        \epsilon_b}

    where :math:`i, j` (resp. :math:`a, b`) indices occupied (resp.
    inoccupied) active spin-orbitals; math:`h_{p, q, r, s}` are the
    2-electron molecular orbital integrals; math:`\epsilon_i` are the
    orbital energies.

    Note:
        The trial parametrization is stored as a dictionary: ``theta[(a,
        b, i, j)]`` describes the parameter :math:`\theta_{a, b}^{i,
        j}`of :math:`a^\dagger_a a^\dagger_b a_i a_j`.

    Args:
        nb_o (int): The number of active spin-orbitals.
        nb_e (int): The number of active electrons.
        int2e (np.ndarray): The 2-electron integrals corrected for
            and reduced to the active space.
        l_ao (List[int]): The list of active spin-orbitals, sorted by
            decreasing NOON
        orbital_energies (List[float]): The vector of spin-orbital
            energies restricted to the active space.
        threshold (float): The numerical threshold used to
            remove smaller terms throughout the execution of the code.

    Return:
        ket_hf_init (np.ndarray): The Hartree-Fock state stored
            as a vector with right-to-left orbitals indexing.
        active_occupied_orbitals (List[int]): The list of the active
            occupied orbitals.
        active_unoccupied_orbitals (List[int]): The list of the active
            unoccupied orbitals.
        theta_init (Dict[Tuple[int], float]): The trial MP2
            parametrization as a dictionary corresponding to the factors
            of each excitation operator (only the terms above
            ``threshold`` are stored.)
    """

    # Construction of the ket vector representing RHF state
    ket_hf_init = np.zeros(nb_o)

    for i in range(nb_e):
        ket_hf_init[i] = 1

    # Convert to integer
    hf_init = BitArray("0b" + "".join([str(int(c)) for c in ket_hf_init])).uint

    active_occupied_orbitals, active_unoccupied_orbitals = construct_active_orbitals(
        nb_e, l_ao
    )

    # Construction of theta_MP2 (to use it as a trial parametrization)
    theta_init = _theta_ab_ij(
        active_occupied_orbitals,
        active_unoccupied_orbitals,
        int2e,
        orbital_energies,
    )
    # Note: At least for initialization, theta_a_i = 0

    return hf_init, theta_init


def construct_active_orbitals(
    nb_e: int, l_ao: List[int]
) -> Tuple[List[int], List[int]]:
    """Construct the active occupied and unoccupied orbitals.

    Args:
        nb_e (int): Number of electrons.
        l_ao (List[int]): The list of active spin-orbitals, sorted by decreasing NOON.

    Returns:
        Tuple[List[int], List[int]]:
            - active occupied orbitals
            - active unoccupied orbitals
    """

    active_occupied_orbitals = []
    active_unoccupied_orbitals = []
    nb_oo = min(l_ao)

    nb_e_left = nb_e - nb_oo
    for i in l_ao:
        if nb_e_left > 0:
            active_occupied_orbitals.append(i)
            nb_e_left -= 1
        else:
            active_unoccupied_orbitals.append(i)

    return active_occupied_orbitals, active_unoccupied_orbitals


def select_excitation_operators(
    noons: List[float],
    active_occupied_orbitals: List[int],
    active_unoccupied_orbitals: List[int],
    max_nb_single_ex: Optional[int] = None,
    max_nb_double_ex: Optional[int] = None,
) -> List[Tuple[int]]:
    r"""Selects the excitation operators to will be used to build the
    cluster operator.

    The UCCSD cluster operator is defined (in normal-ordered form) as:

    .. math::

        T(\theta) = \sum_{a, i} \theta_a^i (a^\dagger_a a_i -
        a^\dagger_i a_a) + \sum_{a > b, i > j} \theta_{a, b}^{i, j}
        (a^\dagger_a a^\dagger_b a_i a_j - a^\dagger_i a^\dagger_j a_a
        a_b)

    where :math:`i, j` (resp. :math:`a, b`) indices occupied (resp.
    unoccupied) spin-orbitals.

    In order to alleviate the computational cost of selecting all the
    excitation operators :math:`a^\dagger_a a_i` and :math:`a^\dagger_a
    a^\dagger_b a_i a_j` (and thus, the full set of parameters), this
    function order the excitation by estimated contribution and selects
    only the best (in accordance with the arguments ``max_nb_single_ex``
    and ``max_nb_double_ex``.)

    Args:
        noons (np.ndarray): The natural orbital occupation numbers
            in an array of size nb_so (number of spatial orbitals.)
        active_occupied_orbitals (List[int]): The list of the active
            occupied orbitals.
        active_unoccupied_orbitals (List[int]): The list of the active
            unoccupied orbitals.
        max_nb_single_ex (Optional[int]): Limit the number of single
            excitation to consider. The number of parameter is the sum
            of this argument and the one below. The default value, 0,
            implies the implementation of UCCD.
        max_nb_double_ex (Optional[int]):  Limit the number of
            double excitation to consider. The number of parameter is
            the sum of this argument and the one above. The default
            value, 3, implies a (partial) implementation of UCC_D.

    Returns:
        l_ex_op (List[Tuple[int]]): The list of of (a, b, i, j) and (a,
            i) tuples describing the excitation operators (without
            Hermitian conjugate, i.e. only excitation from unoccupied to
            occupied orbitals) to consider among the set associated to
            the active orbitals.

    Notes:
        NOONs should be made optional, since it is used only if max_nb_single_ex
        max_nb_double_ex are not None
    """

    l_ex_op = []

    # Determination of NOON variation induced by excitation between 2 orbitals
    var_noons_1e, var_noons_2e = {}, {}

    for a, i in itertools.product(
        active_unoccupied_orbitals[::2], active_occupied_orbitals[::2]
    ):
        # Considering only *singlet* (spin-preserving) single excitation
        var_noons_1e[(a, i)] = noons[a // 2] - noons[i // 2]
        var_noons_1e[(a + 1, i + 1)] = noons[a // 2] - noons[i // 2]

    for n_unocc, a in enumerate(active_unoccupied_orbitals[::1]):
        for b in active_unoccupied_orbitals[n_unocc + 1 :]:
            for n_occ, i in enumerate(active_occupied_orbitals[::1]):
                for j in active_occupied_orbitals[n_occ + 1 :]:

                    if (a % 2 == i % 2 and b % 2 == j % 2) or (
                        a % 2 == j % 2 and b % 2 == i % 2
                    ):
                        var_noons_2e[(b, a, j, i)] = (
                            noons[a // 2]
                            + noons[b // 2]
                            - noons[i // 2]
                            - noons[j // 2]
                        )

        # Considering only *singlet* (spin-preserving) double excitation
        # var_noons_2e[(a + 1, a, i + 1, i)] = noons[a // 2] - noons[i // 2]

    sorted_ex_op_1e = sorted(var_noons_1e, key=var_noons_1e.get)[::-1]
    sorted_ex_op_2e = sorted(var_noons_2e, key=var_noons_2e.get)[::-1]
    # Normal-ordered excitation operators ordered by induced NOON variation.

    # Selection of dominant one-electron excitation operators
    if max_nb_single_ex is None:
        l_ex_op += sorted_ex_op_1e

    else:
        for i in range(max_nb_single_ex):

            if i < len(sorted_ex_op_1e):
                l_ex_op.append(sorted_ex_op_1e[i])

            else:
                break

    # Selection of dominant two-electron excitation operators
    if max_nb_double_ex is None:
        l_ex_op += sorted_ex_op_2e

    else:
        for i in range(max_nb_double_ex):

            if i < len(sorted_ex_op_2e):
                l_ex_op.append(sorted_ex_op_2e[i])

            else:
                break

    return l_ex_op


def get_active_space_hamiltonian(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    noons: List[float],
    nels: int,
    nuclear_repulsion: float,
    threshold_1: Optional[float] = 0.02,
    threshold_2: Optional[float] = 1e-3,
) -> Tuple[ElectronicStructureHamiltonian, List[int], List[int]]:
    r"""Selects the right active space and freezes core electrons
    according to their NOONs :math:`n_i`.

    This function is an implementation of the *Complete Active Space*
    (CAS) approach. It divides orbital space into sets of *active* and
    *inactive* orbitals, the occupation number of the latter remaining
    unchanged during the computation.

    The active space indices are defined as:

    .. math::

        \mathcal{A} = \{i, n_i \in [\varepsilon_2, 2 - \varepsilon_1[\} \cup \{i, n_i \geq 2-\varepsilon_1, 2(i+1)\geq N_e \}

    The inactive occupied orbitals are defined as:

    .. math::

        \mathcal{O} = \{i, n_i \geq 2 -\varepsilon_1, 2(i+1) < N_e \}

    The restriction of the one- and two-body integrals (and update of the core energy)
    is then carried out according to:

    .. math::

        \forall u,v \in \mathcal{A},\; I^{(a)}_{uv} = I_{uv} + \sum_{i\in \mathcal{O}} 2 I_{i,u,v,i} - I_{i,u,i,v}

    .. math::

        \forall u,v,w,x \in \mathcal{A}, I^{(a)}_{uvwx} = I_{uvwx}

    .. math::

        E_\mathrm{core}^{(a)} = E_\mathrm{core} + \sum_{i\in\mathcal{O}} I_{ii} + \sum_{ij\in\mathcal{O}} 2 I_{ijji} - I_{ijij}

    Finally, the one- and two-body integrals :math:`I` are converted to the (spin-resolved)
    one- and two-body coefficients :math:`h`:

    .. math::

        h_{u\sigma, v\sigma'} = I_{u, v} \delta_{\sigma, \sigma'}

    .. math::

        h_{u\sigma_1, v\sigma_2, w\sigma_2', x\sigma_1'} = I_{uvwx} \delta_{\sigma_1, \sigma_1'} \delta_{\sigma_2, \sigma_2'} \left((1-\delta_{\sigma_1,\sigma_2}) + \delta_{\sigma_1,\sigma_2} (1-\delta_{u,v})(1-\delta_{w,x})   \right)

    where the one- and two-body integrals are defined as:

    .. math::

        I_{uv}\equiv(u|h|v)=\int\mathrm{d}r\phi_{u}^{*}(r)T\phi_{v}(r)

    .. math::

        I_{uvwx}\equiv(ux|vw)=\iint\mathrm{d}r_{1}\mathrm{d}r_{2}\phi_{u}^{*}(r_{1})\phi_{x}(r_{1})v(r_{12})\phi_{v}^{*}(r_{2})\phi_{w}(r_{2})

    with :math:`T` (resp. :math:`v`) the one- (resp. two-) body potentials,
    and :math:`\phi_u(r)` is the molecular orbital wavefunction.


    Args:
        one_body_integrals (np.ndarray): 2D array of one-body integrals :math:`I_{uv}`.
        two_body_integrals (np.ndarray): 4D array of two-body integrals :math:`I_{uvwx}`.
        threshold_1 (Optional[float]): The upper threshold :math:`\varepsilon_1` on
            the NOON of an active orbital. Defaults to 0.02.
        noons (List[float]): the natural-orbital occupation numbers :math:`n_i`, sorted
            in descending order (from high occupations to low occupations).
        nels (int): The number of electrons :math:`N_e`.
        nuclear_repulsion (float): value of the nuclear repulsion energy :math:`E_\mathrm{core}`.
        threshold_2 (Optional[float]): The lower threshold :math:`\varepsilon_2` on
            the NOON of an active orbital. Defaults to 0.001.

    Returns:
         Tuple[ElectronicStructureHamiltonian, List[int], List[int]]:
            - the Hamiltonian in active space :math:`H^{(a)}`,
            - the list of indices corresponding to the active orbitals, :math:`\mathcal{A}`,
            - the list of indices corresponding to the occupied orbitals, :math:`\mathcal{O}`.
    """

    active_indices, occupied_indices = select_active_orbitals(
        noons=noons, n_electrons=nels, threshold_1=threshold_1, threshold_2=threshold_2
    )

    core_constant, one_body_as, two_body_as = compute_active_space_integrals(
        one_body_integrals, two_body_integrals, active_indices, occupied_indices
    )

    hpq, hpqrs = convert_to_h_integrals(one_body_as, two_body_as)

    H_active = ElectronicStructureHamiltonian(
        hpq, hpqrs, constant_coeff=nuclear_repulsion + core_constant, do_clean_up=False
    )

    return H_active, active_indices, occupied_indices


def _compute_init_state(
    n_electrons: int,
    noons: List[float],
    orbital_energies: List[float],
    hpqrs: np.ndarray,
) -> Tuple[List[Hamiltonian], int, List[int], List[int]]:
    r"""Find initial guess using Møller-Plesset perturbation theory.

    The trial parametrization is efficiently improved upon the
    Hartree-Fock solution (which would set every initial parameter to
    zero) thanks to the following formula identifying the UCC parameters
    in the Møller-Plesset (MP2) solution :

    .. math::

        \theta_a^i = 0

    .. math::

        \theta_{a, b}^{i, j} = \frac{h_{a, b, i, j} -
        h_{a, b, j, i}}{\epsilon_i + \epsilon_j -\epsilon_a -
        \epsilon_b}

    where :math:`h_{p, q, r, s}` is the 2-electron molecular orbital integral,
    and :math:`\epsilon_i` is the orbital energy.

    Args:
        n_electrons (int): the number of active electrons of the system.
        noons (List[float]): the natural-orbital occupation numbers
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy).
        orbital_energies (List[float]): the energies of the molecular orbitals
            :math:`\epsilon_i` (doubled due to spin degeneracy).
        hpqrs (np.ndarray): the 4D array of (active) two-body integrals :math:`h_{pqrs}`.

    Returns:
        Tuple[List[int], int, List[int], List[int]]:
            - the list of initial coefficients :math:`\{\theta_{a}^{i}, a \in \mathcal{I}', i \in \mathcal{O}' \} \cup \{\theta_{ab}^{ij}, a>b, i>j, a,b \in \mathcal{I}', i,j \in \mathcal{O}'\}`,
            - the integer corresponding to the occupation of the Hartree-Fock solution,
            - the active occupied orbitals indices,
            - the active unoccupied orbitals indices.

    """

    active_size = len(noons)

    (ket_hf_init, theta_init,) = _init_uccsd(
        active_size, n_electrons, hpqrs, list(range(active_size)), orbital_energies
    )

    actives_occupied_orbitals, actives_unoccupied_orbitals = construct_active_orbitals(
        n_electrons, list(range(active_size))
    )

    exc_op_list = select_excitation_operators(
        noons, actives_occupied_orbitals, actives_unoccupied_orbitals
    )
    theta_list = [
        theta_init[op_index] if op_index in theta_init else 0
        for op_index in exc_op_list
    ]

    return (
        theta_list,
        ket_hf_init,
        actives_occupied_orbitals,
        actives_unoccupied_orbitals,
    )


def guess_init_params(
    two_body_integrals: np.ndarray,
    n_electrons: int,
    noons: List[float],
    orbital_energies: List[float],
) -> List[float]:
    """Find initial parameters using Møller-Plesset perturbation theory.

    The trial parametrization is efficiently improved upon the
    Hartree-Fock solution (which would set every initial parameter to
    zero) thanks to the following formula identifying the UCC parameters
    in the Møller-Plesset (MP2) solution :

    .. math::

        \theta_a^i = 0

    .. math::

        \theta_{a, b}^{i, j} = \frac{h_{a, b, i, j} -
        h_{a, b, j, i}}{\epsilon_i + \epsilon_j -\epsilon_a -
        \epsilon_b}

    where :math:`h_{p, q, r, s}` is the 2-electron molecular orbital integral,
    and :math:`\epsilon_i` is the orbital energy.

    Args:

        two_body_integrals (np.ndarray): 4D array of two-body integrals :math:`I_{uvwx}`.
        n_electrons (int): The number of active electrons of the system.
        noons (List[float]): the natural-orbital occupation numbers
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy).
        orbital_energies (List[float]): The energies of the molecular orbitals
            :math:`\epsilon_i` (doubled due to spin degeneracy).

    Returns:
        theta_list (List[float]):
            The list of initial coefficients :math:`\{\theta_{a}^{i}, a \in \mathcal{I}',
            i \in \mathcal{O}' \} \cup \{\theta_{ab}^{ij}, a>b, i>j, a,b \in \mathcal{I}',
            i,j \in \mathcal{O}'\}`,
    """

    noons = _extend_list(noons)
    orbital_energies = _extend_list(orbital_energies)

    hpqrs = _two_body_integrals_to_h(two_body_integrals)
    (
        theta_list,
        _,
        _,
        _,
    ) = _compute_init_state(n_electrons, noons, orbital_energies, hpqrs)

    return theta_list


def get_hf_ket(n_electrons: int, noons: List[float]) -> int:
    """
    Get Hartree-Fock state stored as a vector with right-to-left orbitals indexing.

    Args:
        n_electrons (int): The number of active electrons of the system.
        noons (List[float]): the natural-orbital occupation numbers
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy).

    Returns:
        int: Hartree-Fock state.
    """

    noons = _extend_list(noons)
    hf_init = _hf_ket(n_electrons, noons)

    return hf_init


def _hf_ket(n_electrons: int, noons: List[float]) -> int:
    """Construct the Hartree-Fock state bitstring.

    Args:
        n_electrons (int): The number of active electrons of the system.
        noons (List[float]): the natural-orbital occupation numbers
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy).

    Returns:
        int: Hartree-Fock state.
    """

    ket_hf_init = np.zeros(len(noons))
    for i in range(n_electrons):
        ket_hf_init[i] = 1

    hf_init = BitArray("0b" + "".join([str(int(c)) for c in ket_hf_init])).uint

    return hf_init


def get_cluster_ops(n_electrons: int, noons: List[float]) -> List[Hamiltonian]:
    """Compute the cluster operators.

    Args:
        n_electrons (int): The number of active electrons of the system.
        noons (List[float]): the natural-orbital occupation numbers
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy).

    Returns:
        List[Hamiltonian]: The list of cluster operators :math:`\{T_{a}^{i}, a \in
            \mathcal{I}', i \in \mathcal{O}' \} \cup \{T_{ab}^{ij}, a>b, i>j, a,b \in
            \mathcal{I}', i,j \in \mathcal{O}'\}`

    """

    noons = _extend_list(noons)

    (
        actives_occupied_orbitals,
        actives_unoccupied_orbitals,
    ) = construct_active_orbitals(n_electrons, list(range(len(noons))))

    active_size = len(noons)

    exc_op_list = select_excitation_operators(
        noons, actives_occupied_orbitals, actives_unoccupied_orbitals
    )

    cluster_list = build_cluster_operator(exc_op_list, active_size)

    return cluster_list


def _extend_list(lst: List[Any]) -> List[Any]:
    """Extend a list by cloning every element.

    Args:
        lst (List[Any]): List to extend

    Returns:
        extended_lst (List[Any]): Extended list
    """

    extended_lst = []
    for idx in range(len(lst)):
        extended_lst.extend((lst[idx], lst[idx]))

    return extended_lst
