# -*- coding: utf-8 -*-
"""
Nearest-neighbour matchgates custom gate definition
"""

from typing import List, Optional, Tuple
import numpy as np
import scipy.optimize

from qat.lang.AQASM import QRoutine, RZ, AbstractGate
from qat.core import default_gate_set
from qat.core.circuit_builder.matrix_util import gen_x, gen_y, gen_z


def generalized_pauli_gate(pauli_str: str):
    """Generate an AbstractGate corresponding to the input Pauli string, and add it to the default gate set.

    Args:
        pauli_str (str): Pauli rotation the gate should perform.

    Example:

        .. run-block:: python

            from qat.lang.AQASM import Program
            from qat.core.console import display
            from qat.fermion.matchgates import generalized_pauli_gate

            # Define Pauli rotation gate
            RXYZ = generalized_pauli_gate("XYZ")

            # Define Program and Variable
            prog = Program()
            reg = prog.qalloc(RXYZ.arity)
            theta = prog.new_var(float, "\\theta")

            prog.apply(RXYZ(theta), *reg)

            circ = prog.to_circ()
            display(circ, batchmode=True)

    """

    def _gen_r(var: float):
        """
        Generator for the generalized Pauli rotation gate.
        """

        def _optimized_pauli_kron(pauli_1, pauli_2):
            """
            Faster version of the Kronecker product. Defined as such for optimization purpose.
            """
            return np.einsum("ik,jl", pauli_1, pauli_2).reshape((len(pauli_1) * 2,) * 2)

        # Map string to Pauli matrices
        map_dict = {"X": gen_x(), "Y": gen_y(), "Z": gen_z()}

        # Guess arity depending on Pauli string
        arity = len(pauli_str)

        # Compute the Kronecker products if needed
        kron = map_dict[pauli_str[0]]
        for idx in range(len(pauli_str) - 1):
            kron = _optimized_pauli_kron(kron, map_dict[pauli_str[idx + 1]])

        # Build the final computed gate
        gate = np.eye(2**arity) * np.cos(var / 2) - 1j * kron * np.sin(var / 2)

        return gate

    r = AbstractGate(f"R{pauli_str}", [float], arity=len(pauli_str), matrix_generator=_gen_r)

    return r


# Generate required Pauli gate
RXX = generalized_pauli_gate("XX")
RXY = generalized_pauli_gate("XY")
RYX = generalized_pauli_gate("YX")
RYY = generalized_pauli_gate("YY")
RZZ = generalized_pauli_gate("ZZ")

gate_set = default_gate_set()
gate_set.add_signature(RXX)
gate_set.add_signature(RXY)
gate_set.add_signature(RYX)
gate_set.add_signature(RYY)
gate_set.add_signature(RZZ)


def MG_chain_routine(angles, slater: Optional[bool] = False, zz_angle: Optional[bool] = None) -> QRoutine:
    """
    Routine made of the 4 (resp.2) successive NN MG defining the building
    block of the gaussian state (resp. slater determinant) preparation circuit.

    Notes:
        Order corresponds to AB BA [AA BB] in _make_index_pair_list
    """

    q_rout = QRoutine()

    q_rout.apply(RYY(2 * angles[0]), 0, 1)
    q_rout.apply(RXX(-2 * angles[1]), 0, 1)
    if zz_angle is not None:
        q_rout.apply(RZZ(-2 * zz_angle), 0, 1)
    if not slater:
        q_rout.apply(RYX(2 * angles[2]), 0, 1)
        q_rout.apply(RXY(-2 * angles[3]), 0, 1)

    return q_rout


def gaussian_state_prep_routine(nb_fermionic_modes, theta, slater: Optional[bool] = False) -> QRoutine:
    """
    Routine to prepare a gaussian state associated with an even number of fermionic modes.
    Corresponds to U_Bog in DD.

    Note:
        theta : M Rz angles + (M/2) * { even * lenm + odd * lenm   },
        with lenm = 2 for slater, 4 for not slater.
    """
    m = nb_fermionic_modes

    if len(theta) != (2 * m**2 - m if not slater else m**2):
        raise Exception("Theta doesn't have the correct length!")

    lenm = 2 if slater else 4

    q_rout = QRoutine()

    for i in range(m):
        q_rout.apply(RZ(-2 * theta[i]), i)  # = exp(j theta Z)

    for k in range(m // 2):

        for j in range(m // 2):

            offset = m + lenm * (m - 1) * k + lenm * j
            angles = theta[offset : offset + lenm]

            q_rout.apply(MG_chain_routine(angles, slater), 2 * j, 2 * j + 1)

        for j in range(m // 2 - 1):

            offset = m + lenm * (m - 1) * k + lenm * m // 2 + lenm * j
            angles = theta[offset : offset + lenm]

            q_rout.apply(MG_chain_routine(angles, slater), 2 * j + 1, 2 * j + 2)

    return q_rout


def LDCA_cycle_routine(nb_fermionic_modes, theta_mg, theta_rzz, slater: Optional[bool] = False) -> QRoutine:
    r"""
    Low Depth Circuit Ansatz building block.

    It implements the block called :math:`U_{\mathrm{Var MG}}^{NN(l)}` in the reference article.

    .. note::
        Instead of taking only one array :math:`\\theta` as argument to define gates, it takes
        :math:`\\theta_{\mathrm{MG}}` that corresponds to the same gates as the gaussian state prep circuit and
        :math:`\\theta_{\mathrm{RZZ}}` that corresponds to the insertions of :math:`RZZ` gates.

        The goal is to make it easier when we do VQE to start from the set of parameters
        preparing the gaussian state corresponding to our Hamiltonian, :math:`\\theta_{\mathrm{quad}}`
        (which is the same length as :math:`\\theta_{\mathrm{MG}}`), and to pick randomly or set to 0 the
        initial parameters of the :math:`RZZ` gates.

    Args:
        nb_fermionic_mode (int): (even) number of fermionic modes
        theta_mg (numpy array): angles parametrizing the matchgates corresponding to :math:`U_{\mathrm{Bog}}`
        theta_rzz (numpy array): angles parametrizing the :math:`RZZ` gates inserted in :math:`U_{\mathrm{Bog}}`
        slater (bool, optional): whether to only include excitation-preserving rotations.
                                 Defaults to False.

    Returns:
        QRoutine
    """

    M = nb_fermionic_modes

    if len(theta_mg) != (2 * M**2 - 2 * M if not slater else M**2 - M):
        raise Exception("theta_mg doesn't have the correct length!")

    if len(theta_rzz) != M // 2 * (M - 1):
        raise Exception("theta_rzz doesn't have the correct length!")

    lenm = 2 if slater else 4
    q_rout = QRoutine()

    for k in range(M // 2):

        for j in range(M // 2):  # even rotations

            offset = lenm * (M - 1) * k + lenm * j
            angles_mg = theta_mg[offset : offset + lenm]

            q_rout.apply(
                MG_chain_routine(angles_mg, slater, theta_rzz[(M - 1) * k + j]),
                2 * j,
                2 * j + 1,
            )

        for j in range(M // 2 - 1):  # odd rotations

            offset = lenm * (M - 1) * k + lenm * M // 2 + lenm * j
            angles_mg = theta_mg[offset : offset + lenm]

            q_rout.apply(
                MG_chain_routine(angles_mg, slater, theta_rzz[(M - 1) * k + M // 2 + j]),
                2 * j + 1,
                2 * j + 2,
            )

    return q_rout


def LDCA_routine(
    nb_fermionic_modes,
    ncycles,
    theta,
    theta_gaussian=None,
    slater: Optional[bool] = False,
) -> QRoutine:
    """
    Full LDCA routine

    Args:
        nb_fermionic_modes (int)(even) number of fermionic modes
        ncycles (int): number of LDCA cycles
        theta (numpy array): angles parametrizing the MG gates
        theta_gaussian (numpy array): angles parametrizing U_Bog (unused)
        slater (bool, optional): whether to only include excitation-preserving rotations.
                                 Defaults to False.

    Returns:
        QRoutine

    Notes:
        theta = [ M angles for RZ | angles for RXX/YY/YX/XY | angles for RZZ]
    """
    M = nb_fermionic_modes
    lenm = 4 if not slater else 2
    nb_mg_cycles = M // 2

    l_theta_mg = ncycles * nb_mg_cycles * lenm * (M - 1)
    l_theta_rzz = ncycles * nb_mg_cycles * (M - 1)

    if len(theta) != M + l_theta_mg + l_theta_rzz:
        raise Exception("Theta doesn't have the correct length!")

    if theta_gaussian is not None:

        if len(theta_gaussian) != M + l_theta_mg // ncycles:
            raise Exception("Theta-gaussian doesn't have the correct length!")

    q_rout = QRoutine()

    for i in range(M):

        # q_rout.apply(X, i)
        q_rout.apply(RZ(-2 * theta[i]), i)

    # first M+l_theta_mg angles correspond to same params as U_Bog
    theta_mg = theta[M : M + l_theta_mg]

    # remainder: RZZ angles
    theta_rzz = theta[M + l_theta_mg :]

    pointeur_mg = 0
    pointeur_rzz = 0

    for _ in range(ncycles):

        theta_mg_cycle = theta_mg[pointeur_mg : pointeur_mg + nb_mg_cycles * lenm * (M - 1)]
        theta_rzz_cycle = theta_rzz[pointeur_rzz : pointeur_rzz + nb_mg_cycles * (M - 1)]
        q_rout.apply(
            LDCA_cycle_routine(nb_fermionic_modes, theta_mg_cycle, theta_rzz_cycle, slater),
            list(range(M)),
        )
        pointeur_mg += nb_mg_cycles * lenm * (M - 1)
        pointeur_rzz += nb_mg_cycles * (M - 1)

    return q_rout


def make_gen(a, b, M):
    """Construct antimsymmetric generator :math:`M_{ab}`
    2M x 2M matrix with 1 at (a,b), -1 at b,a
    """
    res = np.zeros((2 * M, 2 * M))
    res[a, b] = 1
    res[b, a] = -1

    return res


def build_generator_old(theta, a, b, M):
    """
    Returns:
        :math:`exp(\thetaM_{ab})`

    Notes:
        Same as build_generator, but slower and less accurate
    """
    M = make_gen(a, b, M)

    return scipy.linalg.expm(theta * M)


def build_generator(theta, a, b, M):
    """
    Returns:
        :math:`exp(\thetaM_{ab})`
    """
    res = np.identity(2 * M)
    res += np.sin(theta) * make_gen(a, b, M)
    res[a, a] += np.cos(theta) - 1
    res[b, b] += np.cos(theta) - 1

    return res


def _make_index_pair_list(M, slater: Optional[bool] = False):

    if slater:
        majorana_components = [(0, 1), (1, 0)]  # AB BA

    else:
        majorana_components = [(0, 1), (1, 0), (0, 0), (1, 1)]  # AB BA AA BB

    index_pair_list = [(i, M + i) for i in range(M)]

    for k in range(M // 2):

        for j in range(M // 2):  # even

            for A, B in majorana_components:
                index_pair_list.append((M * A + 2 * j, M * B + 2 * j + 1))

        for j in range(M // 2 - 1):  # odd

            for A, B in majorana_components:
                index_pair_list.append((M * A + 2 * j + 1, M * B + 2 * j + 2))

    return index_pair_list


def prepare_R_matrix(nb_fermionic_modes: int, theta: np.ndarray, slater: Optional[bool] = False) -> np.ndarray:
    r"""
    Routine to prepare a gaussian state associated with an even number of fermionic modes.

    ..math::
        R = \prod_a \exp(2 \theta_a M_a)

    Args:
        nb_fermionic_modes (int): Number of fermionic modes M.
        theta (np.ndarray): The angles :math:`\theta_a`.
        slater (Optional[bool]): Whether to only include excitation-preserving rotations. Defaults to False.

    Returns:
        np.ndarray: R, a 2Mx2M matrix.
    """

    M = nb_fermionic_modes

    if len(theta) != (2 * M**2 - M if not slater else M**2):
        raise Exception("Theta doesn't have the correct length!")

    # r = exp(2*theta*G), with G generator
    # G: Generator
    # Gtilde(mu nu, i j) = G(M*mu + i, M*nu + j)

    index_pair_list = _make_index_pair_list(M, slater)

    mat = np.identity(2 * M)

    for ind, (a, b) in enumerate(index_pair_list):
        mat = np.dot(build_generator(2 * theta[ind], a, b, M), mat)

    return mat


def prepare_R_matrix_gradient(nb_fermionic_modes: int, theta: np.ndarray, slater: Optional[bool] = False) -> List[np.ndarray]:
    r"""
    Routine to compute gradient of product :math:`R = prod_i r_i(\\theta_i)`

    Returns:
        List[np.ndarray]: List of :math:`2Mx2M` matrices
    """

    M = nb_fermionic_modes

    if len(theta) != (2 * M**2 - M if not slater else M**2):
        raise Exception("Theta doesn't have the correct length!")

    index_pair_list = _make_index_pair_list(M, slater)

    grad_list = []
    for theta_ind in range(len(index_pair_list)):

        mat = np.identity(2 * M)

        for ind, (a, b) in enumerate(index_pair_list):

            if ind == theta_ind:
                mat = np.dot(2 * make_gen(a, b, M), mat)

            mat = np.dot(build_generator(2 * theta[ind], a, b, M), mat)

        grad_list.append(mat)

    return grad_list


def find_R_angles(
    r_target: np.ndarray,
    theta0: Optional[np.ndarray] = None,
    slater: Optional[bool] = False,
    use_gradient: Optional[bool] = False,
    method: Optional[str] = "COBYLA",
    options: Optional[dict] = None,
    verbose: Optional[bool] = False,
) -> Tuple[np.ndarray, float]:
    r"""
    Compute the angles corresponding to the product decomposition

    .. math::
        R(\theta) = \prod_i r_i(theta_i)

    by minimization of the objective function

    .. math::
        f(\theta) = - \mathrm{tr} R^T R(\theta) / (2 n_\mathrm{qbits}).

    Args:
        r_target (np.ndarray): The target rotation matrix R (2Mx2M matrix)
            (in Majorana notation).
        theta0 (Optional[np.ndarray]): The initial guess for the angles.
            Defaults to None (in which case initialization to zero).
        slater (Optional[bool]): Whether to only include excitation-preserving rotations. Defaults to False.
        use_gradient (Optional[bool]): Whether to use a gradient-based optimizer.
            Defaults to False.
        method (Optional[str]): What scipy.optimize.minimize optimizer to use.
            Defaults to COBYLA.
        options (Optional[dict]): What options to pass to the optimizer.
            Defaults to None.
        verbose (Optional[bool]): For verbose output. Defaults to False.

    Returns:
        np.ndarray, float: The list of angles and the value of the objective function

    """
    nqbits = r_target.shape[0] // 2

    def func(theta):

        R_mat = prepare_R_matrix(nqbits, theta, slater)

        return -np.trace(r_target.T.dot(R_mat)) / (2 * nqbits)

    def der(theta):

        der_res = np.zeros_like(theta)
        grad_list = prepare_R_matrix_gradient(nqbits, theta, slater)

        for idx, _ in enumerate(grad_list):
            der_res[idx] = -np.trace(r_target.T.dot(grad_list[idx])) / (2 * nqbits)

        return der_res

    if theta0 is None:
        nparams = 2 * nqbits**2 - nqbits if not slater else nqbits**2
        theta0 = [0.0 for _ in range(nparams)]

    if verbose:
        print("f(x0) = ", func(theta0))

    if not use_gradient:
        res = scipy.optimize.minimize(func, theta0, method=method, options=options)
    else:
        res = scipy.optimize.minimize(func, theta0, method=method, options=options, jac=der)

    if verbose:
        print(res)

    return res.x, res.fun


def get_nn_rotation_angles(
    h: np.ndarray,
    slater: Optional[bool] = False,
    theta0: Optional[np.ndarray] = None,
    method: Optional[str] = "COBYLA",
    options: Optional[dict] = None,
    use_gradient: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> Tuple[np.ndarray, float]:
    r"""
    Compute the rotation angles corresponding to the preparation of the ground state of a quadratic Hamiltonian with Givens
    rotations. The obtained angles can be used as initial angle values in the LDCA circuit ansatz (completed with zeros), so as to
    generate a Slater determinant as an initial trial state for VQE.
        
    The ground state of a non-interacting (=quadratic) Hamiltonian :math:`H=\sum \limits_{pq} h_{pq} c^{\dagger}_p c_q` belongs to
    the class of so-called *gaussian states*. Such states are similar to the well-known Slater determinants, except that they are not
    bound to have a fixed number of particles.

    Similarly to Slater determinants, they can be prepared easily as computational basis states by carrying out a suitable
    single-particle orbital rotation, namely a transformation :math:`\mathcal{R}(\boldsymbol{\theta}) \in SO(2M)` acting on the
    Majorana operators (with :math:`M` the number of fermionic modes):
    :math:`\boldsymbol{\gamma'} = \mathcal{R}(\boldsymbol{\theta}) \boldsymbol{\gamma}` with
    :math:`\boldsymbol{\theta} \in \mathbb{R}^d` the rotation's parameters and :math:`\boldsymbol{\gamma}`
    the vector of :code:`2M` Majorana operators:

    .. math::

        \gamma_k = c^{\dagger}_k + c_k ; \gamma_{k+M} = -i(c^{\dagger}_k - c_k) ; 0 \leq k \leq M-1.

    The rotation :math:`\mathcal{R}` can be decomposed into *Givens rotations* that only act either on local modes or on
    nearest-neighbor modes:

    .. math::

        \mathcal{R}(\boldsymbol{\theta}) =\prod \limits_{j=1}^d r_j(\theta_j)

    which yield a circuit well-suited for linear qubit topology. The target ground state can thus be obtained with a quantum routine as
    :math:`|\psi_0\rangle = U_{\mathcal{R}(\boldsymbol{\theta})} |\psi_{\mathrm{ref}}\rangle`.

    This function computes the Givens rotations' angles associated with to ground state of the non-interacting Hamiltonian.

    Args:
        h (np.ndarray): One-particle matrix (in p-h notation)
        slater (Optional[bool]): Whether to assume number conservation (True)
            or not (False). Defaults to False.
        theta0 (Optional[np.ndarray]): Initial guess for angles. Defaults to None
        method (Optional[str]): Classical optimization method. Defaults to COBYLA
        options (Optional[dict]): Options to be passed to the optimizer. Defaults to None
        use_gradient (Optional[bool]): Whether to use a gradient-based optimizer.
            Defaults to False
        verbose (Optional[bool]): Verbose output. Defaults to False.

    Returns:
        Tuple[np.ndarray, float]: The angles and distance to target total rotation

    """

    _, r = np.linalg.eigh(h)
    nqbits = h.shape[0]

    r_majo = np.block([[r.real, r.imag], [-r.imag, r.real]])

    if slater:
        theta0 = [0.1 for _ in range(nqbits**2)]
    else:
        theta0 = [0.1 for _ in range(2 * nqbits**2 - nqbits)]

    if verbose:
        print("theta0=", theta0)

    theta, cost_res = find_R_angles(
        r_majo,
        theta0,
        slater=slater,
        method=method,
        options=options,
        use_gradient=use_gradient,
        verbose=verbose,
    )

    return theta, cost_res


def nb_params_LDCA(nb_fermionic_modes: int, ncycles: int, slater: Optional[bool] = False):
    """
    Computes the number of variational parameters carried by the LDCA circuit.

    Args:
        nb_fermionic_modes (int): Number of qubits (=number of fermionic modes). Must be even.
        ncycles (int): Number of LDCA cycles.
        slater (Optional[bool]): Whether to only include excitation-preserving rotations. Defaults to False.

    Returns:
        int: Number of parameters of the associated LDCA circuit

    """

    lenm = 4 if not slater else 2

    m = nb_fermionic_modes
    nb_mg_cycles = m // 2

    l_theta_mg = ncycles * nb_mg_cycles * lenm * (m - 1)
    l_theta_rzz = ncycles * nb_mg_cycles * (m - 1)

    return m + l_theta_mg + l_theta_rzz
