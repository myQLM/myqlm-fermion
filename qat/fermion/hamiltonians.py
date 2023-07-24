# -*- coding: utf-8 -*-
"""
Hamiltonian wrappers and constructors
"""

from typing import List, Optional, Union
from copy import deepcopy
import itertools
from numbers import Number
import numpy as np
import scipy.sparse as sp

from qat.core import Observable, Term
from qat.core.variables import BaseArithmetic

from .util import init_creation_ops, dag
from .fermion_algebra import FermionicTerm, normal_order_fermionic_term, order_qubits


PAULI_MATS = {
    "X": [[0, 1], [1, 0]],
    "I": [[1, 0], [0, 1]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
}


def _transform_to_normal_order(terms, nbqbits):
    """
    Transform fermionic terms of a FermionHamiltonian to normally ordered fermionic terms.
    """

    # Initialize empty hamiltonian
    ordered_hamiltonian = FermionHamiltonian(nbqbits, terms=[])

    # Add ordered terms to Hamiltonian
    for term in terms:
        new_term = normal_order_fermionic_term(term)

        for element in new_term:

            if new_term:
                ordered_hamiltonian += FermionHamiltonian(nbqbits, terms=[element], normal_order=False)

    return ordered_hamiltonian.terms


def _preprocess_terms(terms, nbqbits, normal_order):
    """Preprocess input terms into FermionicTerms with or without normal ordering.

    Args:
        terms (List[Term]): List of fermionic terms.
        nbqbits (int): Number of qbits of the hamiltonian which terms are being processed
        normal_order (bool): If the FermionicTerms sould be normally ordered.
    """
    # Converts to FermionicTerm if needed
    terms = terms if isinstance(terms[0], FermionicTerm) else [FermionicTerm.from_term(term) for term in terms]

    # Ensure normal ordering of the fermionic terms
    return _transform_to_normal_order(terms, nbqbits) if normal_order else terms


class SpinHamiltonian(Observable):
    r"""
    Implementation of a spin Hamiltonian.

    Args:
        nqbits (int): the total number of qubits
        terms (List[Term]): the list of terms
        constant_coeff (float): constant term

    Attributes:
        nbqbits (int): the total number of qubits
        terms (List[Term]): the list of terms
        constant_coeff (float): constant term
        matrix (np.ndarray): the corresponding matrix (None by default, can be set by calling get_matrix method)

    Example:

        .. run-block:: python

            from qat.core import Term
            from qat.fermion import SpinHamiltonian

            hamiltonian = SpinHamiltonian(2, [Term(0.3, "X", [0]), Term(-0.4, "ZY", [0, 1])])

            print(f"H = {hamiltonian}")
            print(f"H matrix: {hamiltonian.get_matrix()}")

    """

    def __init__(
        self,
        nqbits: int,
        terms: List[Term],
        constant_coeff: float = 0.0,
    ):

        self.matrix = None

        super(SpinHamiltonian, self).__init__(
            nqbits,
            pauli_terms=terms,
            constant_coeff=constant_coeff,
        )

        # Fast consistency check on the first term inputted.
        self._fast_consistency_check()

    def __add__(self, other):
        res = super().__add__(other)
        return SpinHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __radd__(self, other):
        res = super().__radd__(other)
        return SpinHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __sub__(self, other):
        res = super().__sub__(other)
        return SpinHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __rsub__(self, other):
        res = super().__rsub__(other)
        return SpinHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __mul__(self, other):
        res = super().__mul__(other)
        return SpinHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __rmul__(self, other):
        res = super().__rmul__(other)
        return SpinHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __or__(self, other):
        return self * other - other * self

    def _fast_consistency_check(self):
        """
        Assert that the first term inputted does not contain fermionic operators.

        Note: To avoid large overhead, only the first term is verified.
        """

        if self.terms:
            for op in ["C", "c"]:
                if op in self.terms[0].op:
                    raise TypeError("SpinHamiltonian does not support fermionic operators. Please use FermionHamiltonian instead.")

    def dag(self) -> "SpinHamiltonian":
        """Compute the conjugate transpose of the Hamiltonian.

        Returns:
            SpinHamiltonian: Conjugate transpose of the SpinHamiltonian operator

        """

        return SpinHamiltonian(
            self.nbqbits,
            [Term(np.conj(term.coeff), term.op, term.qbits) for term in self.terms],
            np.conj(self.constant_coeff),
        )

    def get_matrix(self, sparse: bool = False) -> np.ndarray:
        r"""
        This function returns the matrix corresponding to :math:`H` in the computational basis.

        Args:
            sparse (Optional[bool]): Whether to return in sparse representation.
            Defaults to False.

        Returns:
            np.ndarray: The matrix of the SpinHamiltonian.

        Warning:
            This method should not be used if the SpinHamiltonian is too large.

        """

        id_type = sp.identity if sparse else np.identity

        # Precompute needed spin ops
        op_list = {}

        for term in self.terms:
            for op, qb in zip(term.op, term.qbits):

                if (op, qb) not in op_list.items():

                    if op != "I":
                        try:
                            op_list[(op, qb)] = self._make_spin_op(op, qb, self.nbqbits, sparse)

                        except Exception as exc:
                            raise ValueError("An error has occured during the Kronecker product.") from exc

        final_matrix = 0

        for term in self.terms:
            matrix = id_type(2**self.nbqbits, dtype="complex")

            for op, qb in zip(term.op, term.qbits):

                if op != "I":
                    matrix = matrix.dot(op_list[(op, qb)])

            final_matrix += term.coeff * matrix

        final_matrix += self.constant_coeff * id_type(2**self.nbqbits)
        self.matrix = final_matrix

        return final_matrix

    @staticmethod
    def _make_spin_op(op: str, qb: int, nqbits: int, sparse: bool) -> Union[np.ndarray, sp.bsr_matrix]:
        """Build spin operator.

        Args:
            op (str): Pauli string.
            qb (int): Target qubit.
            nqbits (int): Number of qubits.
            sparse (bool): If a sparse matrix should be returned.

        Returns:
            Union[np.ndarray, sp.bsr_matrix]: Matrix of the spin operator.
        """

        id_type = sp.identity if sparse else np.identity
        m_type = sp.csr_matrix if sparse else np.array
        kron_op = sp.kron if sparse else np.kron

        if qb == 0:
            return kron_op(
                m_type(PAULI_MATS[op]),
                id_type(2 ** (nqbits - 1), dtype="complex"),
            )

        if qb == nqbits - 1:
            return kron_op(
                id_type(2 ** (nqbits - 1), dtype="complex"),
                m_type(PAULI_MATS[op]),
            )

        kron = kron_op(
            id_type(2**qb, dtype="complex"),
            kron_op(
                m_type(PAULI_MATS[op]),
                id_type(2 ** (nqbits - qb - 1), dtype="complex"),
            ),
        )
        return kron

    def copy(self):
        """Deepcopy the current class.

        Returns:
            :class:`~qat.fermion.hamiltonians.SpinHamiltonian`: Copy of the SpinHamiltonian.
        """
        return deepcopy(self)


class FermionHamiltonian(Observable):
    r"""
    Implementation of a fermionic Hamiltonian.

    Args:
        nqbits (int): The total number of qubits
        terms (List[Term]): The list of terms
        constant_coeff (float): Constant term
        normal_order (bool, optional): If the fermionic terms should be normal (or Wick) ordered. Default to True. True is
            recommended always.

    Attributes:
        nbqbits (int): The total number of qubits
        terms (List[Term]): The list of terms
        constant_coeff (float): Constant term.
        matrix (np.ndarray): The corresponding matrix (None by default, can be set by calling get_matrix method).
        normal_order (bool): If the fermionic terms should be normal (or Wick) ordered.

    Note:
        Fermionic Hamiltonians are by default automatically normally ordered.

    Example:

        .. run-block:: python

            from qat.core import Term
            from qat.fermion import FermionHamiltonian

            hamiltonian = FermionHamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
            print(f"H = {hamiltonian}")
            print(f"H matrix: {hamiltonian.get_matrix()}")

    """

    def __init__(
        self,
        nqbits: int,
        terms: List[Term],
        constant_coeff: float = 0.0,
        normal_order: bool = True,
    ):

        self.matrix = None

        if terms:
            terms = _preprocess_terms(terms, nqbits, normal_order)

        super().__init__(
            nqbits,
            pauli_terms=terms,
            constant_coeff=constant_coeff,
        )

    def copy(self):
        """Deepcopy the current class.

        Returns:
            :class:`~qat.fermion.hamiltonians.FermionHamiltonian`: Copy of the FermionHamiltonian.
        """
        return deepcopy(self)

    def __add__(self, other):
        res = super().__add__(other)
        return FermionHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __radd__(self, other):
        res = super().__radd__(other)
        return FermionHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __sub__(self, other):
        res = super().__sub__(other)
        return FermionHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __rsub__(self, other):
        res = super().__rsub__(other)
        return FermionHamiltonian(res.nbqbits, res.terms, res.constant_coeff)

    def __mul__(self, other):
        if isinstance(other, (Number, BaseArithmetic)):
            # double deepcopy of the terms, could be reduced to one by creating a new observable instead
            # of copying the existing one
            new_ham = self.copy()
            terms = [*map(deepcopy, self.terms)]
            for term in terms:
                term.coeff *= other
            new_ham.set_terms(terms)
            new_ham.constant_coeff *= other
            return new_ham

        term_list = []
        for term in self.terms:
            term_list.append(FermionicTerm(term.coeff * other.constant_coeff, term.op, term.qbits))

        for term in other.terms:
            # pylint: disable=E1101
            term_list.append(FermionicTerm(self.constant_coeff * term.coeff, term.op, term.qbits))

        for term1, term2 in itertools.product(self.terms, other.terms):
            term_list.append(term1._term * term2._term)

        fermionic_hamiltonian = FermionHamiltonian(self.nbqbits, terms=term_list, normal_order=True)

        return fermionic_hamiltonian

    def __or__(self, other):
        return self * other - other * self

    def __rmul__(self, other):

        if isinstance(other, (Number, BaseArithmetic)):
            return self * other

        return other * self

    def dag(self) -> "FermionHamiltonian":
        """Compute the conjugate transpose of the Hamiltonian.

        Returns:
            FermionHamiltonian: Conjugate transpose of the Hamiltonian.

        """
        # pylint: disable=E1101
        return FermionHamiltonian(
            self.nbqbits,
            [FermionicTerm(np.conj(term.coeff), term.op, term.qbits) for term in self.terms],
            np.conj(self.constant_coeff),
        )

    def get_matrix(self, sparse: bool = False) -> np.ndarray:
        r"""
        This function returns the matrix corresponding to :math:`H` in the computational basis.

        Args:
            sparse (Optional[bool]): Whether to return in sparse representation.
            Defaults to False.

        Returns:
            numpy.ndarray: The matrix of the FermionHamiltonian.

        Warning:
            This method should not be used if the Hamiltonian is too large.

        """

        ops = {}
        ops["C"] = init_creation_ops(self.nbqbits, sparse=sparse)
        ops["c"] = {ind: dag(c_dag) for ind, c_dag in ops["C"].items()}

        id_type = sp.identity if sparse else np.identity

        final_matrix = 0
        for term in self.terms:

            matrix = id_type(2**self.nbqbits, dtype="complex")

            for op, qb in zip(term.op, term.qbits):
                matrix = matrix.dot(ops[op][qb])

            final_matrix += term.coeff * matrix

        # pylint: disable=E1101
        final_matrix += self.constant_coeff * id_type(2**self.nbqbits)
        self.matrix = final_matrix

        return final_matrix

    def to_spin(self, method: Optional[str] = "jordan-wigner"):
        """Maps the fermionic Hamiltonian to a spin Hamiltonian.

        Args:
            method (str, optional): Method to use for the transformation to a spin representation. Available methods are :

                    - "jordan-wigner" : Jordan-Wigner transform (default),
                    - "bravyi-kitaev" : Bravyi-Kitaev transform,
                    - "parity" : Parity transform.

        Returns:
            :class:`~qat.fermion.hamiltonians.SpinHamiltonian` : Hamiltonian in spin representation.

        """
        # pylint: disable=C0415
        from qat.fermion.transforms import (
            transform_to_jw_basis,
            transform_to_bk_basis,
            transform_to_parity_basis,
        )

        transform_map = {
            "jordan-wigner": transform_to_jw_basis,
            "bravyi-kitaev": transform_to_bk_basis,
            "parity": transform_to_parity_basis,
        }

        return transform_map[method](self)

    def to_electronic(self):
        """Converts a fermionic Hamiltonian to a electronic-structure Hamiltonian. This can be done only if the Hamiltonian
        contains only single and double interaction operators (i.e. only "Cc" and "CCcc" fermionic operators).

        Returns:
            :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian` : Electronic-structure Hamiltonian.

        """

        nqbits = self.nbqbits
        hpq = np.zeros((nqbits,) * 2, dtype="complex")
        hpqrs = np.zeros((nqbits,) * 4, dtype="complex")

        def _fill_tensors(_term):

            indices = _term.qbits

            if _term.op == "Cc":
                hpq[tuple(indices)] += _term.coeff

            if _term.op == "CCcc":
                hpqrs[tuple(indices)] += 2 * _term.coeff

        for term in self.terms:

            if not ((term.op.count("C") == 1 and term.op.count("c") == 1) or (term.op.count("C") == 2 and term.op.count("c") == 2)):
                raise TypeError(
                    "The Hamiltonian contains fermionic operators incompatible with"
                    " a transformation to a electronic-structure Hamiltonian."
                    "The terms need to have either one creation and one annihilation operator,"
                    " or two creation and two annihilation operators"
                )

            if not (term.op == "Cc" or term.op == "CCcc"):
                no_terms = normal_order_fermionic_term(term)  # normal-ordered

                for no_term in no_terms:
                    _fill_tensors(no_term)

            else:
                term = order_qubits(term)
                _fill_tensors(term)

        return ElectronicStructureHamiltonian(hpq, hpqrs)


class ElectronicStructureHamiltonian(FermionHamiltonian):
    r"""
    A container for the electronic-structure Hamiltonian, defined as

    .. math::
        H = \sum_{pq} h_{pq}a_p^\dagger a_q
            + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s
            + c \mathbb{I}

    Args:
        hpq (np.ndarray): Array :math:`h_{pq}`. Must be 2D.
        hpqrs (np.ndarray): Array :math:`h_{pqrs}`. Must be 4D.
        constant_coeff (float): Constant coefficient :math:`c.`

    Attributes:
        hpq (np.ndarray): Array :math:`h_{pq}`.
        hpqrs (np.ndarray): Array :math:`h_{pqrs}`.
        constant_coeff (float): Constant coefficient :math:`c`.

    Example:

        .. run-block:: python

            import numpy as np
            from qat.fermion import ElectronicStructureHamiltonian

            h_pq = 0.2 * np.array([[0, 1], [1, 0]])
            h_pqrs = np.zeros((2, 2, 2, 2))
            h_pqrs[0, 1, 1, 0] = 0.7
            h_pqrs[1, 0, 0, 1] = 0.7
            hamiltonian = ElectronicStructureHamiltonian(h_pq, h_pqrs, -6)

            print(f"H = {hamiltonian}")
            eigvals = np.linalg.eigvalsh(hamiltonian.get_matrix())

            print(f"eigenvalues = {eigvals}")

    """

    TOL = 1e-12

    def __init__(
        self,
        hpq: np.ndarray,
        hpqrs: np.ndarray = None,
        constant_coeff: float = 0.0,
    ):

        if hpqrs is None:
            hpqrs = np.zeros((hpq.shape[0], hpq.shape[0], hpq.shape[0], hpq.shape[0]))

        self._hpq = hpq
        self._hpqrs = hpqrs

        terms = self._get_fermionic_terms()

        super(ElectronicStructureHamiltonian, self).__init__(self._hpq.shape[0], terms, constant_coeff)
        
    @property
    def hpq(self):
        """hpq getter.

        Returns:
            np.ndarray: hpq matrix
        """
        return self._hpq
    
    @hpq.setter
    def hpq(self, value):
        """
        hpq setter.
        """
        self._hpq = value
        terms = self._get_fermionic_terms()
        # pylint: disable=E1101
        super(ElectronicStructureHamiltonian, self).__init__(self._hpq.shape[0], terms, self.constant_coeff)

    @property
    def hpqrs(self):
        """hpqs getter.

        Returns:
            np.ndarray: hpqrs matrix
        """
        return self._hpqrs
    
    @hpqrs.setter
    def hpqrs(self, value):
        """
        hpqrs getter.
        """
        self._hpqrs = value
        terms = self._get_fermionic_terms()
        # pylint: disable=E1101
        super(ElectronicStructureHamiltonian, self).__init__(self._hpq.shape[0], terms, self.constant_coeff)

    def dag(self) -> "ElectronicStructureHamiltonian":
        """Compute the conjugate transpose of the Hamiltonian.

        Returns:
            ElectronicStructureHamiltonian: Conjugate transpose of the Hamiltonian.

        """
        # pylint: disable=E1101
        return ElectronicStructureHamiltonian(np.conj(self.hpq), np.conj(self.hpqrs), np.conj(self.constant_coeff))
    
    def copy(self):
        """Deepcopy the current class.

        Returns:
            :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian`: Copy of the ElectronicStructureHamiltonian.
        """
        return deepcopy(self)
    
    def __add__(self, other):

        # pylint: disable=E1101
        return ElectronicStructureHamiltonian(
            self.hpq + other.hpq,
            self.hpqrs + other.hpqrs,
            self.constant_coeff + other.constant_coeff,
        )

    def _get_fermionic_terms(self) -> List[Term]:
        """Get the FermionicHamiltonian terms from current ElectronicStructureHamiltonian.

        Returns:
            terms (List[Term]): Fermionic terms of the ElectronicStructureHamiltonian
        """

        terms = []
        for i, j in itertools.product(range(self._hpq.shape[0]), range(self._hpq.shape[1])):

            if abs(self._hpq[i, j]) > ElectronicStructureHamiltonian.TOL:
                terms.append(Term(self._hpq[i, j], "Cc", [i, j]))

        for i, j, k, l in itertools.product(
            range(self._hpqrs.shape[0]),
            range(self._hpqrs.shape[1]),
            range(self._hpqrs.shape[2]),
            range(self._hpqrs.shape[3]),
        ):

            if abs(self._hpqrs[i, j, k, l]) > ElectronicStructureHamiltonian.TOL:
                terms.append(FermionicTerm(0.5 * self._hpqrs[i, j, k, l], "CCcc", [i, j, k, l]))

        return terms

    def to_fermion(self) -> FermionHamiltonian:
        """Convert current ElectronicStructureHamiltonian to a FermionHamiltonian.

        Returns:
            FermionHamiltonian: Fermionic Hamiltonian.
        """

        terms = self._get_fermionic_terms()

        # pylint: disable=E1101
        return FermionHamiltonian(self.hpq.shape[0], terms, self.constant_coeff)


def make_anderson_model(u: float, mu: float, v: np.ndarray, epsilon: np.ndarray) -> ElectronicStructureHamiltonian:
    r"""
    Returns the canonical second quantized form

    .. math::
        H_{\mathrm{CSQ}} = \sum_{p,q} h_{pq} f_p^\dagger f_q + \frac{1}{2}\sum_{p,q,r,s} h_{pqrs} f_p^\dagger f_q^\dagger f_r f_s

    of a single impurity coupled with :math:`n_b` bath modes Anderson model Hamiltonian

    .. math::
        H_{\mathrm{SIAM}} = U c_{\uparrow}^\dagger c_{\uparrow} c_{\downarrow}^\dagger c_{\downarrow} - \mu(c_{\uparrow}^\dagger c_{\uparrow}+c_{\downarrow}^\dagger c_{\downarrow})
        + \sum_{i=1..n_b} \sum_{\sigma=\uparrow,\downarrow} V_i (c_{\sigma}^\dagger a_{i,\sigma} + \mathrm{h.c.}) \\
        + \sum_{i=1..n_b} \sum_{\sigma=\uparrow,\downarrow} \epsilon_i a_{i,\sigma}^\dagger a_{i,\sigma}.

    Args:
        U (float): Coulomb repulsion intensity.
        mu (float): Chemical potential.
        V (np.ndarray): Tunneling energies. This vector has the same size as the number of bath mode.
        epsilon (np.ndarray): Bath modes energies. This vector has the same size as the number of bath mode.

    Returns:
        :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian` object constructed from :math:`h_{pq}` (matrix of size
        :math:`(2n_b+2) \times (2n_b+2)`) and :math:`h_{pqrs}` (4D tensor with size :math:`2n_b+2` in each dimension)

    .. note::
        Convention:
        :math:`f_0` corresponds to :math:`c_{\uparrow}` (annihilation in the 'up' mode of the impurity),
        :math:`f_1` corresponds to :math:`c_{\downarrow}` (annihilation in the 'down' mode of the impurity),
        :math:`f_2` corresponds to :math:`a_{1,\uparrow}` (annihilation in the 'up' mode of the 1st bath mode),
        :math:`f_3` corresponds to :math:`a_{1,\downarrow}` (annihilation in the 'down' mode of the 1st bath mode),
        and so on.

    """

    # number of bath modes
    n_b = len(v)
    if len(epsilon) != n_b:
        raise Exception("Error: The bath modes energies vector must be the same size as the tunneling energies vector.")

    # number of fermionic (annihilation) operators f
    fermop_number = 2 * n_b + 2

    h_pq = np.zeros((fermop_number, fermop_number))
    h_pqrs = np.zeros((fermop_number, fermop_number, fermop_number, fermop_number))

    # single spin localized on the impurity
    h_pq[0, 0] = -mu
    h_pq[1, 1] = -mu

    # bath modes terms
    for i in range(0, n_b):
        h_pq[2 * (i + 1), 2 * (i + 1)] += epsilon[i]
        h_pq[2 * (i + 1) + 1, 2 * (i + 1) + 1] += epsilon[i]

    # hopping terms
    for i in range(0, n_b):
        h_pq[0, 2 * (i + 1)] += v[i]
        h_pq[2 * (i + 1), 0] += v[i]
        h_pq[1, 2 * (i + 1) + 1] += v[i]
        h_pq[2 * (i + 1) + 1, 1] += v[i]

    # Coulomb repulsion when the impurity is occupied by two spins. The minus sign comes from the commutation we need to do in the
    # U-term to get the operators in the right order.
    h_pqrs[0, 1, 0, 1] = -u
    h_pqrs[1, 0, 1, 0] = -u

    return ElectronicStructureHamiltonian(h_pq, h_pqrs)


def make_embedded_model(
    u: float,
    mu: float,
    D: np.ndarray,
    lambda_c: np.ndarray,
    t_loc: Optional[np.ndarray] = None,
    int_kernel: Optional[np.ndarray] = None,
    grouping: Optional[str] = "spins",
) -> ElectronicStructureHamiltonian:
    r"""
    Returns the canonical second quantized form

    .. math::
        H_{\mathrm{CSQ}} = \sum_{p,q} h_{pq} f_p^\dagger f_q + \frac{1}{2}\sum_{p,q,r,s} h_{pqrs} f_p^\dagger f_q^\dagger f_r f_s + c\mathbb{I}

    of an embedded hamiltonian

    .. math::
        H_{\mathrm{emb}} = U \sum \limits_{i,j,k,l=1}^{2M} I_{ijkl} f^{\dagger}_i f_j f^{\dagger}_k f_l
                       - \mu \sum \limits_{i=1}^{M} f^{\dagger}_{i} f_{j}
                       + \sum \limits_{i, j=1}^{M} t^{\mathrm{loc}}_{ij} f^{\dagger}_i f_j \\
                       + \sum \limits_{i,j=1}^{M} (D_{ij} f^{\dagger}_{i} f_{M+j} + \mathrm{h.c.}) \\
                       + \sum \limits_{i,j=1}^{M} \lambda^c_{ij} f_{M+i} f^{\dagger}_{M+j}

    where :math:`M` is the number of orbitals (imp+bath). Indices here correspond to the spin-orbitals ordering referred to as
    'cluster' (see below).

    Args:
        U (float): Onsite repulsion on impurity sites.
        mu (float): Chemical potential.
        D (np.ndarray): Hopping matrix (i.e. hybridization) between the correlated orbitals and the uncorrelated bath.
        lambda_c (np.ndarray): Hopping matrix of the uncorrelated sites.
        t_loc (Optional[np.ndarray]): Hopping matrix of the correlated sites.
        int_kernel (Optional[np.ndarray]): Array :math:`I` with 1 at position :math:`i, j, k, l` where :math:`U` must be put
                                        (conv. for associated term: :math:`c^{\dagger}c^{\dagger}cc`). Defaults to None,
                                        in which case :math:`U` is put before terms :math:`c^{\dagger}_{2i}c^{\dagger}_{2i+1}c_{2i}c_{2i+1}, i=1..M/2` if grouping is 'clusters', :math:`c^{\dagger}_{i}c^{\dagger}_{i+M}c_{i}c_{i+M}, i=1..M/2` if grouping is 'spins'.
                                        This array must be a 4D array.
        grouping (Optional[str]): Defines how spin-orbitals indices are ordered (see below), defaults to 'spins'.

    Returns:
        :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian`

    The two grouping strategies are the following:

    - **"clusters"**: the first :math:`M` orbitals SO are :math:`(\uparrow, \mathrm{imp}_0), (\downarrow, \mathrm{imp}_0),..., (\uparrow, \mathrm{imp}_{M-1}), (\downarrow, \mathrm{imp}_{M-1})` and the last :math:`M` orbitals are bath orbitals with similar ordering.
    - **"spins"**: the first :math:`M` orbitals are :math:`(\uparrow, \mathrm{imp}_0), (\uparrow, \mathrm{imp}_1), ..., (\uparrow, \mathrm{bath}_{M-2}), (\uparrow, \mathrm{bath}_{M-1})` and the last :math:`M` orbitals are down orbitals with similar ordering.

    """

    M = np.shape(lambda_c)[0]  # number of SO in each cluster (imp and bath) = 2*cluster size

    h_pq = np.zeros((2 * M, 2 * M), dtype=np.complex_)

    if int_kernel is None:
        h_pqrs = np.zeros((2 * M, 2 * M, 2 * M, 2 * M))
    else:
        h_pqrs = -u * int_kernel

    const_coeff = 0

    for i in range(M):

        h_pq[i, i] += -mu  # energy of impurity levels

        for j in range(M):
            # energy of uncorrelated levels
            h_pq[i + M, j + M] += -lambda_c[i, j]
            h_pq[i, j + M] += D[i, j]  # hopping between the two clusters
            h_pq[j + M, i] += np.conj(D[i, j])  # hopping
            if t_loc is not None:
                h_pq[i, j] += t_loc[i, j]

        const_coeff += lambda_c[i, i]

    if grouping == "spins":
        perm_mat = np.zeros((2 * M, 2 * M))  # permutation matrix, beware: goes from spin ord to cluster ord!

        for i in range(M):
            perm_mat[2 * i, i] = 1
            perm_mat[2 * i + 1, i + M] = 1

        if int_kernel is None and u != 0:
            for i in range(M // 2):
                a = ind_clusters_ord(2 * i, M)
                b = ind_clusters_ord(2 * i + 1, M)
                h_pqrs[a, b, a, b] = -u
                h_pqrs[b, a, b, a] = -u

        h_pq = np.einsum("ap, pq, bq", perm_mat, h_pq, perm_mat)

    elif grouping == "clusters":
        if int_kernel is None and u != 0:
            for i in range(M // 2):
                h_pqrs[2 * i, 2 * i + 1, 2 * i, 2 * i + 1] = -u  # minus sign comes from the def. of hpqrs: term c_dag c_dag c c
                h_pqrs[2 * i + 1, 2 * i, 2 * i + 1, 2 * i] = -u
    else:
        print("Grouping must be either clusters or spins.")

    return ElectronicStructureHamiltonian(h_pq, h_pqrs, const_coeff)


def ind_clusters_ord(i: int, n_orb: int) -> int:
    """
    Computes the indice with cluster-ordering (up, dn, ..., up, dn)_imp(up, dn, ..., up, dn)_bath of spin-orbital of index
    i in spin-ordering
    (up_imp1, up_imp2, ..., up_bath1, ..., up_bathM)(dn_imp1, dn_imp2, ..., dn_bath1, ..., dn_bathM)

    Args:
        i (int): Indice (with spin-ordering) of the spin-orbital we want to compute the indice in cluster-ordering of.
        n_orb (int): Number of orbitals (imp+bath).

    """

    if i < n_orb:
        return 2 * i

    try:
        assert i < 2 * n_orb

    except AssertionError:
        print("index must be lesser than 2*n_orb")

    else:
        return 2 * i - (2 * n_orb - 1)


def ind_spins_ord(i: int, n_orb: int) -> int:
    """
    Computes the indice with spin-ordering
    (up_imp1, up_imp2, ..., up_bath1, ..., up_bathM)(dn_imp1, dn_imp2, ..., dn_bath1, ..., dn_bathM) of spin-orbital of index
    i in cluster-ordering (up, dn, ..., up, dn)_imp(up, dn, ..., up, dn)_bath

    Args:
        i (int): Indice (with cluster-ordering) of the spin-orbital we want to compute the indice in spin-ordering of.
        n_orb (int): Number of orbitals (imp+bath).
    """

    idx_spins_ord = (i % 2 - 1) * (-i // 2) + (i % 2) * ((i - 1) // 2 + n_orb)

    return idx_spins_ord


def make_hubbard_model(t_mat: np.ndarray, U: float, mu: float) -> ElectronicStructureHamiltonian:
    r"""Constructs Hubbard model

    .. math::
        H = \sum_{ij,\sigma} (t_{ij} -  mu \delta_{ij}) c^\dagger_i c_j + U \sum_i n_{i\uparrow} n_{i \downarrow} 

    Args:
        t_mat (np.ndarray): Hopping matrix (n_sites x n_sites). t_mat may have diagonal terms contributing 
                            to the chemical potential on each site.
        U (float): Hubbard U.
        mu (float): Reference chemical potential.

    Returns:
        ElectronicStructureHamiltonian: The Hubbard Hamiltonian.

    Notes:
        Spin-orbital labeling convention: :math:`i \equiv (k, \sigma) = 2 k + \sigma`
        with :math:`i`: site index and :math:`\sigma`: spin index.

    """
    nqbit = 2 * t_mat.shape[0]

    hpq = np.zeros((nqbit, nqbit))

    for i, j in itertools.product(range(t_mat.shape[0]), range(t_mat.shape[1])):

        for sig in [0, 1]:
            hpq[2 * i + sig, 2 * j + sig] = t_mat[i, j]

    for i in range(t_mat.shape[0]):

        for sig in [0, 1]:
            hpq[2 * i + sig, 2 * i + sig] += -mu

    hpqrs = np.zeros((nqbit, nqbit, nqbit, nqbit))

    for i in range(t_mat.shape[0]):

        for sig in [0, 1]:

            hpqrs[2 * i + sig, 2 * i + 1 - sig, 2 * i + sig, 2 * i + 1 - sig] = -U

    return ElectronicStructureHamiltonian(hpq=hpq, hpqrs=hpqrs)


def make_tot_density_op(n_sites: int) -> ElectronicStructureHamiltonian:
    r"""Construct total density operator.

    .. math::
        N = \sum_{i \sigma} n_{i\sigma}

    Args:
        n_sites (int): Number of sites.

    Returns:
        ElectronicStructureHamiltonian: The total density operator N.

    """
    nqbit = 2 * n_sites
    tot_density = np.zeros((nqbit, nqbit))

    for i in range(n_sites):
        for sig in [0, 1]:
            tot_density[2 * i + sig, 2 * i + sig] = 1

    tot_density_op = ElectronicStructureHamiltonian(hpq=tot_density)

    return tot_density_op
