# -*- coding: utf-8 -*-
"""
Deprecated UCC tools
"""

from warnings import warn
from typing import Callable, List, Optional, Tuple, Dict
import itertools
from bitstring import BitArray
import numpy as np

from qat.core import Term
from qat.lang.AQASM import QRoutine, X

from .ucc import (
    _build_cluster_operator,
    _construct_active_orbitals,
    _init_uccsd,
    _theta_ab_ij,
    tobin,
    select_active_orbitals,
    compute_active_space_integrals,
    convert_to_h_integrals,
    get_hf_ket,
)
from ..hamiltonians import SpinHamiltonian, FermionHamiltonian, ElectronicStructureHamiltonian
from ..trotterisation import make_spin_hamiltonian_trotter_slice


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
        noons (List[float]): The natural orbital occupation numbers
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

    """

    l_ex_op = []

    # Determination of NOON variation induced by excitation between 2 orbitals
    var_noons_1e, var_noons_2e = {}, {}

    for a, i in itertools.product(active_unoccupied_orbitals[::2], active_occupied_orbitals[::2]):

        # Considering only *singlet* (spin-preserving) single excitation
        var_noons_1e[(a, i)] = noons[a // 2] - noons[i // 2]
        var_noons_1e[(a + 1, i + 1)] = noons[a // 2] - noons[i // 2]

    for n_unocc, a in enumerate(active_unoccupied_orbitals[::1]):
        for b in active_unoccupied_orbitals[n_unocc + 1 :]:
            for n_occ, i in enumerate(active_occupied_orbitals[::1]):
                for j in active_occupied_orbitals[n_occ + 1 :]:

                    if (a % 2 == i % 2 and b % 2 == j % 2) or (a % 2 == j % 2 and b % 2 == i % 2):
                        var_noons_2e[(b, a, j, i)] = noons[a // 2] + noons[b // 2] - noons[i // 2] - noons[j // 2]

    # Normal-ordered excitation operators ordered by induced NOON variation.
    sorted_ex_op_1e = sorted(var_noons_1e, key=var_noons_1e.get)[::-1]
    sorted_ex_op_2e = sorted(var_noons_2e, key=var_noons_2e.get)[::-1]

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


def build_ucc_ansatz(cluster_ops: List[SpinHamiltonian], ket_hf: int, n_steps: Optional[int] = 1) -> Callable:
    r"""Builds the parametric state preparation circuit implementing the
    provided cluster operator.

    The returned function maps :math:`\vec{\theta}` to a QRoutine
    describing :math:`Q` such as:

    .. math::

        Q \vert \vec{0} \rangle
            &= \vert \mathrm{UCC} (\vec{\theta}) \rangle \\
            &= e^{T(\vec{\theta})} \vert \mathrm{HF}\rangle

    Args:
        cluster_ops (List[SpinHamiltonian]): The cluster operators iT (note the i factor)
        ket_hf (int): The Hartree-Fock state in integer representation
        n_steps(int): Number of trotter steps

    Returns:
        Lambda (Callable): The parametric state preparation implementing the UCCSD method, theta -> QRoutine

    Warning:
        Deprecated
    """

    nqbits = cluster_ops[0].nbqbits
    n_ops = len(cluster_ops)

    # Convert to string
    ket_hf_init_sp = [int(c) for c in tobin(ket_hf, nqbits)]

    # Construction of the QRoutine corresponding to U such as |HF> = U |0>
    qrout_hf = QRoutine(arity=nqbits)
    for j in range(nqbits):
        if int(ket_hf_init_sp[j]) == 1:
            qrout_hf.apply(X, j)

    def qroutwparam(theta):
        r"""Returns the QRoutine describing :math:`Q` such as:
        .. math::

            Q \vert \vec{0} \rangle
                &= \vert \mathrm{UCC} (\vec{\theta}) \rangle \\
                &= e^{T(\vec{\theta})} \vert \mathrm{HF}\rangle

        Args:
            theta (lst(float)): The trial parametrization as a
                dictionary corresponding to the factors of each
                excitation operator.

        Returns:
            qrout_uccsd (QRoutine): The QRoutine implementing
                :math: `\vert UCCSD(\vec{\theta})\rangle` with the
                fixed parameter set ``theta`` given in input.

        """
        qrout_uccsd = QRoutine(arity=nqbits)
        qubit_range = list(range(nqbits))

        assert n_ops == len(theta), "received {0} cluster operators and {1} variational parameters".format(n_ops, len(theta))

        # Application of qrout_hf
        qrout_uccsd.apply(qrout_hf, qubit_range)

        # Construction of the QRoutine corresponding to exp T = exp (sum_op_index theta_op_index * ex_op_op_index)
        for i in range(n_steps):
            terms = []

            for angle, cluster_op in zip(theta, cluster_ops):
                # op_index = (i, a) or (i, j, a, b)

                for term in cluster_op.terms:
                    assert isinstance(term.coeff, (float, complex))

                    coeff = term.coeff
                    if isinstance(term.coeff, complex):
                        assert term.coeff.imag < 1e-13
                        coeff = term.coeff.real

                    coeff = angle * term.coeff
                    terms.append(Term(coeff, term.op, term.qbits))

            # QRoutine implementation
            cluster_op_obs = SpinHamiltonian(nqbits, terms)
            qrout_expt = make_spin_hamiltonian_trotter_slice(cluster_op_obs)

            # Approx to exp(-i O), with O = i T
            qrout_uccsd.apply(qrout_expt, qubit_range[: qrout_expt.arity])

        return qrout_uccsd

    return qroutwparam


def get_cluster_ops(
    active_noons: List[float],
    actives_occupied_orbitals: List[int],
    actives_unoccupied_orbitals: List[int],
) -> List[SpinHamiltonian]:
    r"""Build the cluster operator.

    The UCCSD cluster operator is defined (in normal-ordered form) as:

    .. math::

        T(\theta) = \sum_{a, i} \theta_a^i (c^\dagger_a c_i -
        c^\dagger_i c_a) + \sum_{a > b, i > j} \theta_{a, b}^{i, j}
        (c^\dagger_a c^\dagger_b c_i c_j - c^\dagger_i c^\dagger_j c_a
        c_b)

    where :math:`i, j \in \mathcal{O}'`, and :math:`a, b \in \mathcal{I}'`,
    with :math:`\mathcal{I}'` (resp. :math:`\mathcal{O}'`) the list of inoccupied
    (resp. occupied) orbitals (doubled due to spin degeneracy)

    Args:
        active_noons (list<float>): The natural-orbital occupation numbers.
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy).
        active_orb_energies (list<float>): The energies of the molecular orbitals
            :math:`\epsilon_i` (doubled due to spin degeneracy).
        hpqrs (np.array): The 4D array of (active) two-body integrals :math:`h_{pqrs}`.

    Returns:
        List[FermionHamiltonian]: The list of cluster operators :math:`\{T_{a}^{i}, a \in \mathcal{I}', i \in \mathcal{O}' \} \cup \{T_{ab}^{ij}, a>b, i>j, a,b \in \mathcal{I}', i,j \in \mathcal{O}'\}`
    """

    warn(
        "This get_cluster_ops function is deprecated.",
        stacklevel=2,
    )

    active_size = len(active_noons)

    exc_op_list = select_excitation_operators(active_noons, actives_occupied_orbitals, actives_unoccupied_orbitals)

    cluster_list = _build_cluster_operator(exc_op_list, active_size)

    return cluster_list


def guess_init_state(
    n_active_els: int,
    active_noons: List[float],
    active_orb_energies: List[float],
    hpqrs: np.ndarray,
) -> Tuple[List[float], int, List[int], List[int]]:
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
        n_active_els (int): The number of active electrons of the system
        active_noons (list<float>): The natural-orbital occupation numbers
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy)
        active_orb_energies (List[float]): The energies of the molecular orbitals
            :math:`\epsilon_i` (doubled due to spin degeneracy)
        hpqrs (np.ndarray): The 4D array of (active) two-body integrals :math:`h_{pqrs}`

    Returns:
        List[float], int, List[int], List[int]:
        - the list of initial coefficients :math:`\{\theta_{a}^{i}, a \in \mathcal{I}', i \in \mathcal{O}' \} \cup \{\theta_{ab}^{ij}, a>b, i>j, a,b \in \mathcal{I}', i,j \in \mathcal{O}'\}`,
        - the integer corresponding to the occupation of the Hartree-Fock solution,
        - the actives occupied orbitals indices,
        - the actives unoccupied orbitals indices.

    """
    warn(
        "This guess_init_state function is deprecated.",
        stacklevel=2,
    )

    active_size = len(active_noons)

    ket_hf_init = get_hf_ket(n_active_els, active_size)
    theta_init = _init_uccsd(n_active_els, hpqrs, list(range(active_size)), active_orb_energies)

    actives_occupied_orbitals, actives_unoccupied_orbitals = _construct_active_orbitals(n_active_els, list(range(active_size)))

    exc_op_list = select_excitation_operators(active_noons, actives_occupied_orbitals, actives_unoccupied_orbitals)
    theta_list = [theta_init[op_index] if op_index in theta_init else 0 for op_index in exc_op_list]

    return (
        theta_list,
        ket_hf_init,
        actives_occupied_orbitals,
        actives_unoccupied_orbitals,
    )


def init_uccsd(
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
        l_ao (list(int)): The list of active spin-orbitals, sorted by
            decreasing NOON
        orbital_energies (List[float]): The vector of spin-orbital
            energies restricted to the active space.

    Returns:
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

    # Construction of the lists of active occupied and unoccupied orbitals
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

    # Construction of theta_MP2 (to use it as a trial parametrization)
    theta_init = _theta_ab_ij(
        active_occupied_orbitals,
        active_unoccupied_orbitals,
        int2e,
        orbital_energies,
    )
    # Note: At least for initialization, theta_a_i = 0

    return hf_init, active_occupied_orbitals, active_unoccupied_orbitals, theta_init


def get_cluster_ops_and_init_guess(
    n_active_els: int,
    active_noons: List[float],
    active_orb_energies: List[float],
    hpqrs: np.ndarray,
) -> Tuple[List[SpinHamiltonian], List[float], int]:
    r"""Build the cluster operator and find initial guess using Møller-Plesset
    perturbation theory.

    The UCCSD cluster operator is defined (in normal-ordered form) as:

    .. math::

        T(\theta) = \sum_{a, i} \theta_a^i (c^\dagger_a c_i -
        c^\dagger_i c_a) + \sum_{a > b, i > j} \theta_{a, b}^{i, j}
        (c^\dagger_a c^\dagger_b c_i c_j - c^\dagger_i c^\dagger_j c_a
        c_b)

    where :math:`i, j \in \mathcal{O}'`, and :math:`a, b \in \mathcal{I}'`,
    with :math:`\mathcal{I}'` (resp. :math:`\mathcal{O}'`) the list of inoccupied
    (resp. occupied) orbitals (doubled due to spin degeneracy)

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
        n_active_els (int): the number of active electrons of the system.
        active_noons (List[float]): the natural-orbital occupation numbers.
            :math:`n_i`, sorted in descending order (from high occupations
            to low occupations) (doubled due to spin degeneracy)
        active_orb_energies (List[float]): the energies of the molecular orbitals
            :math:`\epsilon_i` (doubled due to spin degeneracy).
        hpqrs (np.ndarray): the 4D array of (active) two-body integrals :math:`h_{pqrs}`.

    Returns:
        cluster_list (List[FermionHamiltonian]): List of cluster operators :math:`\{T_{a}^{i}, a
            \in \mathcal{I}', i \in \mathcal{O}' \} \cup \{T_{ab}^{ij}, a>b, i>j, a,b \in
            \mathcal{I}', i,j \in \mathcal{O}'\}`.
        theta_list (List[float]): List of initial coefficients :math:`\{\theta_{a}^{i}, a
        \in \mathcal{I}', i \in \mathcal{O}' \} \cup \{\theta_{ab}^{ij}, a>b, i>j, a,b
        \in \mathcal{I}', i,j \in \mathcal{O}'\}`.
        ket_hf_init (int): Integer corresponding to the occupation of the Hartree-Fock solution.

    """
    warn(
        "This guess_init_state function is deprecated.",
        stacklevel=2,
    )

    active_size = len(active_noons)

    # find theta_init (MP2)
    ket_hf_init, as_occ, as_unocc, theta_init = init_uccsd(
        active_size, n_active_els, hpqrs, list(range(active_size)), active_orb_energies
    )

    exc_op_list = select_excitation_operators(active_noons, as_occ, as_unocc)
    cluster_list = _build_cluster_operator(exc_op_list, active_size)
    theta_list = [theta_init[op_index] if op_index in theta_init else 0 for op_index in exc_op_list]

    return cluster_list, theta_list, ket_hf_init


# def get_active_space_hamiltonian(
#     one_body_integrals: np.ndarray,
#     two_body_integrals: np.ndarray,
#     nels: int,
#     nuclear_repulsion: float,
#     noons: List[float],
#     threshold_1: Optional[float] = 0.02,
#     threshold_2: Optional[float] = 1e-3,
# ) -> Tuple[ElectronicStructureHamiltonian, List[int], List[int]]:
#     r"""Selects the right active space and freezes core electrons
#     according to their NOONs :math:`n_i`.

#     This function is an implementation of the *Complete Active Space*
#     (CAS) approach. It divides orbital space into sets of *active* and
#     *inactive* orbitals, the occupation number of the latter remaining
#     unchanged during the computation.

#     The active space indices are defined as:

#     .. math::

#         \mathcal{A} = \{i, n_i \in [\varepsilon_2, 2 - \varepsilon_1[\} \cup \{i, n_i \geq 2-\varepsilon_1, 2(i+1)\geq N_e \}

#     The inactive occupied orbitals are defined as:

#     .. math::

#         \mathcal{O} = \{i, n_i \geq 2 -\varepsilon_1, 2(i+1) < N_e \}

#     The restriction of the one- and two-body integrals (and update of the core energy)
#     is then carried out according to:

#     .. math::

#         \forall u,v \in \mathcal{A},\; I^{(a)}_{uv} = I_{uv} + \sum_{i\in \mathcal{O}} 2 I_{i,u,v,i} - I_{i,u,i,v}

#     .. math::

#         \forall u,v,w,x \in \mathcal{A}, I^{(a)}_{uvwx} = I_{uvwx}

#     .. math::

#         E_\mathrm{core}^{(a)} = E_\mathrm{core} + \sum_{i\in\mathcal{O}} I_{ii} + \sum_{ij\in\mathcal{O}} 2 I_{ijji} - I_{ijij}

#     Finally, the one- and two-body integrals :math:`I` are converted to the (spin-resolved)
#     one- and two-body coefficients :math:`h`:

#     .. math::

#         h_{u\sigma, v\sigma'} = I_{u, v} \delta_{\sigma, \sigma'}

#     .. math::

#         h_{u\sigma_1, v\sigma_2, w\sigma_2', x\sigma_1'} = I_{uvwx} \delta_{\sigma_1, \sigma_1'} \delta_{\sigma_2, \sigma_2'} \left((1-\delta_{\sigma_1,\sigma_2}) + \delta_{\sigma_1,\sigma_2} (1-\delta_{u,v})(1-\delta_{w,x})   \right)

#     where the one- and two-body integrals are defined as:

#     .. math::

#         I_{uv}\equiv(u|h|v)=\int\mathrm{d}r\phi_{u}^{*}(r)T\phi_{v}(r)

#     .. math::

#         I_{uvwx}\equiv(ux|vw)=\iint\mathrm{d}r_{1}\mathrm{d}r_{2}\phi_{u}^{*}(r_{1})\phi_{x}(r_{1})v(r_{12})\phi_{v}^{*}(r_{2})\phi_{w}(r_{2})

#     with :math:`T` (resp. :math:`v`) the one- (resp. two-) body potentials,
#     and :math:`\phi_u(r)` is the molecular orbital wavefunction.


#     Args:
#         one_body_integrals (np.ndarray): 2D array of one-body integrals :math:`I_{uv}`.
#         two_body_integrals (np.ndarray): 4D array of two-body integrals :math:`I_{uvwx}`.
#         threshold_1 (Optional[float]): The upper threshold :math:`\varepsilon_1` on
#             the NOON of an active orbital. Defaults to 0.02.
#         nels (int): The number of electrons :math:`N_e`.
#         nuclear_repulsion (float): value of the nuclear repulsion energy :math:`E_\mathrm{core}`.
#         noons (List[float]): the natural-orbital occupation numbers :math:`n_i`, sorted
#             in descending order (from high occupations to low occupations).
#         threshold_2 (Optional[float]): The lower threshold :math:`\varepsilon_2` on
#             the NOON of an active orbital. Defaults to 0.001.

#     Returns:
#          Tuple[ElectronicStructureHamiltonian, List[int], List[int]]:
#             - the Hamiltonian in active space :math:`H^{(a)}`,
#             - the list of indices corresponding to the active orbitals, :math:`\mathcal{A}`,
#             - the list of indices corresponding to the occupied orbitals, :math:`\mathcal{O}`.
#     """

#     active_indices, occupied_indices = select_active_orbitals(
#         noons=noons, n_electrons=nels, threshold_1=threshold_1, threshold_2=threshold_2
#     )

#     core_constant, one_body_as, two_body_as = compute_active_space_integrals(
#         one_body_integrals, two_body_integrals, active_indices, occupied_indices
#     )

#     hpq, hpqrs = convert_to_h_integrals(one_body_as, two_body_as)

#     H_active = ElectronicStructureHamiltonian(hpq, hpqrs, constant_coeff=nuclear_repulsion + core_constant)

#     return H_active, active_indices, occupied_indices


def get_active_space_hamiltonian(
    one_body_integrals, two_body_integrals, noons, nels, nuclear_repulsion, threshold_1=0.02, threshold_2=1e-3
):
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
        one_body_integrals (np.array): 2D array of one-body integrals :math:`I_{uv}`
        two_body_integrals (np.array): 4D array of two-body integrals :math:`I_{uvwx}`
        noons (list<float>): the natural-orbital occupation numbers :math:`n_i`, sorted
            in descending order (from high occupations to low occupations)
        nels (int): The number of electrons :math:`N_e`.
        nuclear_repulsion (float): value of the nuclear repulsion energy :math:`E_\mathrm{core}`.
        threshold_1 (float, optional): The upper threshold :math:`\varepsilon_1` on
            the NOON of an active orbital. Defaults to 0.02.
        threshold_2 (float, optional): The lower threshold :math:`\varepsilon_2` on
            the NOON of an active orbital. Defaults to 0.001.

    Returns:
        ElectronicStructureHamiltonian, list<int>, list<int>:

        - the Hamiltonian in active space :math:`H^{(a)}`
        - the list of indices corresponding to the active orbitals, :math:`\mathcal{A}`
        - the list of indices corresponding to the occupied orbitals, :math:`\mathcal{O}`

    """
    active_indices, occupied_indices = select_active_orbitals(
        noons=noons, n_electrons=nels, threshold_1=threshold_1, threshold_2=threshold_2
    )

    core_constant, one_body_as, two_body_as = compute_active_space_integrals(
        one_body_integrals, two_body_integrals, active_indices, occupied_indices
    )

    hpq, hpqrs = convert_to_h_integrals(one_body_as, two_body_as)

    H_active = ElectronicStructureHamiltonian(hpq, hpqrs, constant_coeff=nuclear_repulsion + core_constant)

    return H_active, active_indices, occupied_indices
