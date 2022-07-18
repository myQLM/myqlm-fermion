""" Module with containers for common Hamiltonians """
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from enum import Enum
from itertools import product
import scipy.sparse as sp
import numpy as np
import warnings

from qat.core import Observable, Term

from .util import init_creation_ops, dag

PAULI_MATS = {
    "X": [[0, 1], [1, 0]],
    "I": [[1, 0], [0, 1]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
}

""" Module with containers for common Hamiltonians """


PAULI_MATS = {
    "X": [[0, 1], [1, 0]],
    "I": [[1, 0], [0, 1]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
}


class ObservableType(Enum):
    """
    Define the different types of Hamiltonian
    """

    UNDEFINED = 0
    SPIN = 1
    FERMION = 2
    BOSONIC = 3


class Hamiltonian(Observable):
    r"""
    Implementation of a generic hamiltonian.

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

        One can use spin operators :

        .. run-block:: python

            from qat.core import Term
            from qat.fermion import Hamiltonian

            hamiltonian = Hamiltonian(2, [Term(0.3, "X", [0]), Term(-0.4, "ZY", [0, 1])])
            print(f"H = {hamiltonian}")

            # let us print the corresponding matrix representation:
            print(f"H matrix: {hamiltonian.get_matrix()}")

        Or fermionic operators :

        .. run-block:: python

            from qat.core import Term
            from qat.fermion import Hamiltonian

            hamiltonian = Hamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
            print(f"H = {hamiltonian}")
            print(f"H matrix: {hamiltonian.get_matrix()}")

    """

    def __init__(
        self,
        nqbits: int,
        terms: List[Term],
        constant_coeff: float = 0.0,
        do_clean_up: bool = True,
    ):

        self._type = ObservableType.UNDEFINED
        self.matrix = None
        self.do_clean_up = do_clean_up

        super().__init__(
            nqbits,
            pauli_terms=terms,
            constant_coeff=constant_coeff,
            do_clean_up=do_clean_up,
        )

    def __add__(self, other):
        res = super().__add__(other)
        return Hamiltonian(res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up)

    @property
    def htype(self) -> ObservableType:
        """
        Check the type of the Hamiltonian (spin or fermionic).

        Returns:
            ObservableType: Type of the Hamiltonian

        Warning:
            This method should not be used if the Hamiltonian is too large.

        """
        return self._check_hamiltonian_type()

    def dag(self) -> "Hamiltonian":
        """Compute the conjugate transpose of the Hamiltonian.

        Returns:
            Hamiltonian: Conjugate transpose of the Hamiltonian operator

        """
        return Hamiltonian(
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
            numpy.ndarray: The matrix of the Hamiltonian.

        Warning:
            This method should not be used if the Hamiltonian is too large.

        """

        if self.htype is ObservableType.SPIN:
            return self._get_spin_op_matrix(sparse)

        elif self.htype is ObservableType.FERMION:
            return self._get_fermion_op_matrix(sparse)

    def _get_spin_op_matrix(self, sparse: bool) -> Union[np.ndarray, sp.bsr.bsr_matrix]:
        """
        Get the matrix representation of the Hamiltonian of type SPIN.
        """

        def _make_spin_op(op: str, qb: int, nqbits: int, sparse: bool) -> Union[np.ndarray, sp.bsr.bsr_matrix]:

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

        if self.matrix is not None and sp.issparse(self.matrix) == sparse:
            return self.matrix

        id_type = sp.identity if sparse else np.identity

        # Precompute needed spin ops
        op_list = {}

        for term in self.terms:
            for op, qb in zip(term.op, term.qbits):

                if (op, qb) not in op_list.keys():

                    if op != "I":
                        try:
                            op_list[(op, qb)] = _make_spin_op(op, qb, self.nbqbits, sparse)

                        except:
                            print(op, qb, self.nbqbits)
                            raise

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

    def _get_fermion_op_matrix(self, sparse: bool = False) -> Union[np.ndarray, sp.bsr.bsr_matrix]:
        """
        Get the matrix representation of the Hamiltonian of type FERMION.
        """

        if self.matrix is not None and sp.issparse(self.matrix) == sparse:
            return self.matrix

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

        final_matrix += self.constant_coeff * id_type(2**self.nbqbits)
        self.matrix = final_matrix

        return final_matrix

    def _check_hamiltonian_type(self) -> ObservableType:
        """Check the type of the Hamiltonian.

        Raises:
            NotImplementedError: Raise an error if Hamiltonian is in mixed form (i.e. contains fermionic and spin operators)

        Returns:
            ObservableType: Type of the Hamiltonian.
        """

        condition_1, condition_2 = (None,) * 2

        for term in self.terms:
            for operator in term.op:
                condition_1 = ObservableType.SPIN if operator in PAULI_MATS.keys() else condition_1
                condition_2 = ObservableType.FERMION if operator in {"C", "c"} else condition_2

        if all((condition_1, condition_2)) is None:
            return ObservableType.UNDEFINED

        if isinstance(condition_1, ObservableType) and isinstance(condition_2, ObservableType):
            raise NotImplementedError("Hamiltonian with mixed spin-fermion operators are not implemented.")

        else:
            return condition_1 or condition_2

    def to_spin(self, method: Optional[str] = "jordan-wigner"):
        """Transform a fermionic Hamiltonian to a spin Hamiltonian.

        Args:
            method (str, optional): Method to use for the transformation to a spin representation. Available methods are :

                    - "jordan-wigner" : Jordan-Wigner transform (default),
                    - "bravyi-kitaev" : Bravyi-Kitaev transform,
                    - "parity" : Parity transform.

        Returns:
            :class:`~qat.fermion.hamiltonians.Hamiltonian` : Hamiltonian in spin representation.

        """

        from .transforms import (
            transform_to_jw_basis,
            transform_to_bk_basis,
            transform_to_parity_basis,
        )

        transform_map = {
            "jordan-wigner": transform_to_jw_basis,
            "bravyi-kitaev": transform_to_bk_basis,
            "parity": transform_to_parity_basis,
        }

        if self.htype == ObservableType.SPIN:
            warnings.warn("The Hamiltonian is already in spin representation.")
            return self

        return transform_map[method](self)


class ElectronicStructureHamiltonian(Hamiltonian):
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
        do_clean_up: bool = True,
    ):

        if hpqrs is None:
            hpqrs = np.zeros((hpq.shape[0], hpq.shape[0], hpq.shape[0], hpq.shape[0]))

        self.hpq = hpq
        self.hpqrs = hpqrs
        self.do_clean_up = do_clean_up

        terms = []
        for i, j in product(range(hpq.shape[0]), range(hpq.shape[1])):

            if abs(hpq[i, j]) > ElectronicStructureHamiltonian.TOL:
                terms.append(Term(hpq[i, j], "Cc", [i, j]))

        for i, j, k, l in product(
            range(hpqrs.shape[0]),
            range(hpqrs.shape[1]),
            range(hpqrs.shape[2]),
            range(hpqrs.shape[3]),
        ):

            if abs(hpqrs[i, j, k, l]) > ElectronicStructureHamiltonian.TOL:
                terms.append(Term(0.5 * hpqrs[i, j, k, l], "CCcc", [i, j, k, l]))

        super().__init__(hpq.shape[0], terms, constant_coeff, do_clean_up=do_clean_up)

    def __add__(self, other):

        return ElectronicStructureHamiltonian(
            self.hpq + other.hpq,
            self.hpqrs + other.hpqrs,
            self.constant_coeff + other.constant_coeff,
            do_clean_up=self.do_clean_up,
        )


class SpinHamiltonian:
    """Ensures retrocompatibility of old SpinHamiltonian class with new Hamiltonian class"""

    def __new__(
        cls,
        nqbits: int,
        pauli_terms: List[Term],
        constant_coeff: float = 0.0,
        do_clean_up: bool = True,
    ):

        from warnings import warn

        warn(
            "The SpinHamiltonian class is deprecated. Please use the Hamiltonian class instead.",
            stacklevel=2,
        )
        return Hamiltonian(nqbits=nqbits, terms=pauli_terms, constant_coeff=constant_coeff, do_clean_up=do_clean_up)


class FermionHamiltonian(Hamiltonian):
    """Ensures retrocompatibility of old SpinHamiltonian class with new Hamiltonian class"""

    def __new__(cls, *args, **kwargs):
        from warnings import warn

        warn(
            "The FermionHamiltonian class is deprecated. Please use the Hamiltonian class instead.",
            stacklevel=2,
        )
        return Hamiltonian(*args, **kwargs)


def make_anderson_model(U: float, mu: float, V: np.ndarray, epsilon: np.ndarray) -> ElectronicStructureHamiltonian:
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
        :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian` object constructed from :math:`h_{pq}` (matrix of size :math:`(2n_b+2) \times (2n_b+2)`) and :math:`h_{pqrs}` (4D tensor with size :math:`2n_b+2` in each dimension)

    .. note::
        Convention:
        :math:`f_0` corresponds to :math:`c_{\uparrow}` (annihilation in the 'up' mode of the impurity),
        :math:`f_1` corresponds to :math:`c_{\downarrow}` (annihilation in the 'down' mode of the impurity),
        :math:`f_2` corresponds to :math:`a_{1,\uparrow}` (annihilation in the 'up' mode of the 1st bath mode),
        :math:`f_3` corresponds to :math:`a_{1,\downarrow}` (annihilation in the 'down' mode of the 1st bath mode),
        and so on.    

    """

    # number of bath modes
    n_b = len(V)
    if len(epsilon) != n_b:
        raise Exception("Error : The bath modes energies vector must be the same size as the tunneling energies vector.")

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
        h_pq[0, 2 * (i + 1)] += V[i]
        h_pq[2 * (i + 1), 0] += V[i]
        h_pq[1, 2 * (i + 1) + 1] += V[i]
        h_pq[2 * (i + 1) + 1, 1] += V[i]

    # Coulomb repulsion when the impurity is occupied by two spins. The minus sign comes from the commutation we need to do in the U-term to get the operators in the right order.
    h_pqrs[0, 1, 0, 1] = -U
    h_pqrs[1, 0, 1, 0] = -U

    return ElectronicStructureHamiltonian(h_pq, h_pqrs)


def make_embedded_model(
    U: float,
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

    where :math:`M` is the number of orbitals (imp+bath). Indices here correspond to the spin-orbitals ordering referred to as 'cluster' (see below).

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
        h_pqrs = -U * int_kernel

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

        if int_kernel is None and U != 0:
            for i in range(M // 2):
                a = ind_clusters_ord(2 * i, M)
                b = ind_clusters_ord(2 * i + 1, M)
                h_pqrs[a, b, a, b] = -U
                h_pqrs[b, a, b, a] = -U

        h_pq = np.einsum("ap, pq, bq", perm_mat, h_pq, perm_mat)

    elif grouping == "clusters":
        if int_kernel is None and U != 0:
            for i in range(M // 2):
                h_pqrs[2 * i, 2 * i + 1, 2 * i, 2 * i + 1] = -U  # minus sign comes from the def. of hpqrs: term c_dag c_dag c c
                h_pqrs[2 * i + 1, 2 * i, 2 * i + 1, 2 * i] = -U
    else:
        print("Grouping must be either " "clusters" " or " "spins" ".")

    return ElectronicStructureHamiltonian(h_pq, h_pqrs, const_coeff)


def ind_clusters_ord(ind_spins_ord: int, M: int) -> int:
    """
    Computes the indice with cluster-ordering (up, dn, ..., up, dn)_imp(up, dn, ..., up, dn)_bath of spin-orbital of index
    ind_clusters_ord in spin-ordering  (up_imp1, up_imp2, ..., up_bath1, ..., up_bathM)(dn_imp1, dn_imp2, ..., dn_bath1, ..., dn_bathM)

    Args:
        ind_clusters_ord (int): Indice (with spin-ordering) of the spin-orbital we want to compute the indice in cluster-ordering of.
        M (int): Number of orbitals (imp+bath).

    """

    i = ind_spins_ord

    if i < M:
        return 2 * i

    else:

        try:
            assert i < 2 * M

        except AssertionError:
            print("index must be lesser than 2*M")

        else:
            return 2 * i - (2 * M - 1)


def ind_spins_ord(ind_clusters_ord: int, M: int) -> int:
    """
    Computes the indice with spin-ordering (up_imp1, up_imp2, ..., up_bath1, ..., up_bathM)(dn_imp1, dn_imp2, ..., dn_bath1, ..., dn_bathM)
    of spin-orbital of index ind_clusters_ord in cluster-ordering (up, dn, ..., up, dn)_imp(up, dn, ..., up, dn)_bath

    Args:
        ind_clusters_ord (int): Indice (with cluster-ordering) of the spin-orbital we want to compute the indice in spin-ordering of.
        M (int): Number of orbitals (imp+bath).
    """

    i = ind_clusters_ord
    ind_spins_ord = (i % 2 - 1) * (-i // 2) + (i % 2) * ((i - 1) // 2 + M)

    return ind_spins_ord


def make_hubbard_model(t_mat: np.ndarray, U: float, mu: float) -> ElectronicStructureHamiltonian:
    r"""Constructs Hubbard model

    .. math::
        H = \sum_{ij,\sigma} t_{ij} c^\dagger_i c_j + U \sum_i n_{i\uparrow} n_{i \downarrow} - \mu \sum_i n_i

    Args:
        t_mat (np.ndarray): Hopping matrix (n_sites x n_sites).
        U (float): Hubbard U.
        mu (float): Chemical potential.

    Returns:
        ElectronicStructureHamiltonian: The Hubbard Hamiltonian.

    Notes:
        Spin-orbital labeling convention: :math:`i \equiv (k, \sigma) = 2 k + \sigma`
        with :math:`i`: site index and :math:`\sigma`: spin index.

    """
    nqbit = 2 * t_mat.shape[0]

    hpq = np.zeros((nqbit, nqbit))

    for i, j in product(range(t_mat.shape[0]), range(t_mat.shape[1])):

        for sig in [0, 1]:
            hpq[2 * i + sig, 2 * j + sig] = t_mat[i, j]

    for i in range(t_mat.shape[0]):

        for sig in [0, 1]:
            hpq[2 * i + sig, 2 * i + sig] = -mu

    hpqrs = np.zeros((nqbit, nqbit, nqbit, nqbit))

    for i in range(t_mat.shape[0]):

        for sig in [0, 1]:

            hpqrs[2 * i + sig, 2 * i + 1 - sig, 2 * i + sig, 2 * i + 1 - sig] = -U

    return ElectronicStructureHamiltonian(hpq=hpq, hpqrs=hpqrs)


def make_tot_density_op(n_sites: int) -> ElectronicStructureHamiltonian:
    """Construct total density operator.

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
