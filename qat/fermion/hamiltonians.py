""" Module with containers for common Hamiltonians """
from itertools import product
import scipy.sparse as sp
import numpy as np

from qat.core import Observable, Term

from .util import init_creation_ops, dag

PAULI_MATS = {
    "X": [[0, 1], [1, 0]],
    "I": [[1, 0], [0, 1]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
}

""" Module with containers for common Hamiltonians """
from itertools import product
import scipy.sparse as sp
import numpy as np

from enum import Enum

from qat.core import Observable, Term

from .util import init_creation_ops, dag

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
    """
    Implementation of a generic hamiltonian.

    Args:
        nqbits (int): the total number of qubits
        terms (list<Term>): the list of terms
        constant_coeff (float): constant term

    Attributes:
        nbqbits (int): the total number of qubits
        terms (list<Term>): the list of terms
        constant_coeff (float): constant term
        matrix (np.array): the corresponding matrix (None by default,
            can be set by calling get_matrix method)

    Example:

        One can use spin operators :
        .. run-block:: python

            from qat.core import Term
            from qat.fermion import Hamiltonian

            hamiltonian = Hamiltonian(2, [Term(0.3, "X", [0]), Term(-0.4, "ZY", [0, 1])])
            print(hamiltonian)

            # let us print the corresponding matrix representation:
            print("H matrix:", hamiltonian.get_matrix())

        Or fermionic operators :
        .. run-block:: python

            from qat.core import Term
            from qat.fermion import Hamiltonian

            hamiltonian = Hamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
            print("H = ", hamiltonian)

            # let us print the corresponding matrix representation:
            print("H matrix:", hamiltonian.get_matrix())
    """

    def __init__(self, nqbits, terms, constant_coeff=0.0, do_clean_up=True):

        self.type = ObservableType.UNDEFINED
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
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __radd__(self, other):
        res = super().__radd__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __iadd__(self, other):
        res = super().__iadd__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __sub__(self, other):
        res = super().__sub__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __neg__(self, other):
        res = super().__neg__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __mul__(self, other):
        res = super().__mul__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __rmul__(self, other):
        res = super().__rmul__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __truediv__(self, other):
        res = super().__truediv__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    def __itruediv__(self, other):
        res = super().__itruediv__(other)
        return Hamiltonian(
            res.nbqbits, res.terms, res.constant_coeff, do_clean_up=self.do_clean_up
        )

    @property
    def htype(self):
        return self._check_hamiltonian_type()

    def dag(self):
        return Hamiltonian(
            self.nbqbits,
            [Term(np.conj(term.coeff), term.op, term.qbits) for term in self.terms],
            np.conj(self.constant_coeff),
        )

    def get_matrix(self, sparse=False):
        r"""
        This function returns matrix corresponding to :math:`H` in the computational basis

        Args:
            sparse (bool, optional): whether to return in sparse
                representation. Defaults to False.

        Returns:
            numpy.ndarray or sp.bsr.bsr_matrix: The matrix of the Hamiltonian.
        """

        if self.htype is ObservableType.SPIN:
            return self._get_spin_op_matrix(sparse)

        elif self.htype is ObservableType.FERMION:
            return self._get_fermion_op_matrix(sparse)

    def _get_spin_op_matrix(self, sparse):
        def _make_spin_op(op, qb, nqbits, sparse):
            """
            Args:
                op (str): X, Y or Z
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
            return kron_op(
                id_type(2**qb, dtype="complex"),
                kron_op(
                    m_type(PAULI_MATS[op]),
                    id_type(2 ** (nqbits - qb - 1), dtype="complex"),
                ),
            )

        if self.matrix is not None and sp.issparse(self.matrix) == sparse:
            return self.matrix
        # m_type = sp.csr_matrix if sparse else np.array
        # kron_op = sp.kron if sparse else np.kron
        id_type = sp.identity if sparse else np.identity

        # precompute needed spin ops
        op_list = {}
        for term in self.terms:
            for op, qb in zip(term.op, term.qbits):
                if (op, qb) not in op_list.keys():
                    if op != "I":
                        try:
                            op_list[(op, qb)] = _make_spin_op(
                                op, qb, self.nbqbits, sparse
                            )
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

    def _get_fermion_op_matrix(self, sparse=False):

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

    def _check_hamiltonian_type(self):

        condition_1, condition_2 = (None,) * 2

        for term in self.terms:
            for operator in term.op:
                condition_1 = (
                    ObservableType.SPIN
                    if operator in PAULI_MATS.keys()
                    else condition_1
                )
                condition_2 = (
                    ObservableType.FERMION if operator in {"C", "c"} else condition_2
                )

        if all((condition_1, condition_2)) is None:
            return ObservableType.UNDEFINED

        if isinstance(condition_1, ObservableType) and isinstance(
            condition_2, ObservableType
        ):
            raise NotImplementedError(
                "Hamiltonian with mixed spin-fermion operators are not implemented."
            )

        else:
            return condition_1 or condition_2


class ElectronicStructureHamiltonian(Hamiltonian):
    r"""
    A container for the electronic-structure Hamiltonian, defined as

    .. math::
        H = \sum_{pq} h_{pq}a_p^\dagger a_q
            + \frac{1}{2} \sum_{pqrs} h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s
            + c \mathbb{I}

    Args:
        hpq (np.array): 2D array :math:`h_{pq}`
        hpqrs (np.array): 4D array :math:`h_{pqrs}`
        constant_coeff (float): constant coefficient :math:`c`

    Attributes:
        hpq (np.array): 2D array :math:`h_{pq}`
        hpqrs (np.array): 4D array :math:`h_{pqrs}`
        constant_coeff (float): constant coefficient :math:`c`

    Example:

        .. run-block:: python

            import numpy as np
            from qat.fermion import ElectronicStructureHamiltonian

            h_pq = 0.2 * np.array([[0, 1], [1, 0]])
            h_pqrs = np.zeros((2, 2, 2, 2))
            h_pqrs[0, 1, 1, 0] = 0.7
            h_pqrs[1, 0, 0, 1] = 0.7
            hamiltonian = ElectronicStructureHamiltonian(h_pq, h_pqrs, -6)

            print("H = ", hamiltonian)
            eigvals = np.linalg.eigvalsh(hamiltonian.get_matrix())
            print("eigenvalues =", eigvals)
    """

    def __init__(self, hpq, hpqrs=None, constant_coeff=0.0, do_clean_up=True):
        if hpqrs is None:
            hpqrs = np.zeros((hpq.shape[0], hpq.shape[0], hpq.shape[0], hpq.shape[0]))
        self.hpq = hpq
        self.hpqrs = hpqrs
        self.do_clean_up = do_clean_up
        terms = []
        TOL = 1e-12
        for i, j in product(range(hpq.shape[0]), range(hpq.shape[1])):
            if abs(hpq[i, j]) > TOL:
                terms.append(Term(hpq[i, j], "Cc", [i, j]))

        for i, j, k, l in product(
            range(hpqrs.shape[0]),
            range(hpqrs.shape[1]),
            range(hpqrs.shape[2]),
            range(hpqrs.shape[3]),
        ):
            if abs(hpqrs[i, j, k, l]) > TOL:
                terms.append(Term(0.5 * hpqrs[i, j, k, l], "CCcc", [i, j, k, l]))

        super().__init__(hpq.shape[0], terms, constant_coeff, do_clean_up=do_clean_up)

    def __add__(self, other):
        return ElectronicStructureHamiltonian(
            self.hpq + other.hpq,
            self.hpqrs + other.hpqrs,
            self.constant_coeff + other.constant_coeff,
            do_clean_up=self.do_clean_up,
        )


def make_hubbard_model(t_mat, U, mu):
    r"""Construct Hubbard model

    .. math::
        H = \sum_{ij,\sigma} t_{ij} c^\dagger_i c_j + U \sum_i n_{i\uparrow} n_{i \downarrow} - \mu \sum_i n_i

    Args:
        t_mat (np.array): hopping matrix (n_sites x n_sites)
        U (float): Hubbard U
        mu (float): chemical potential

    Returns:
        ElectronicStructureHamiltonian: the Hubbard hamiltonian

    Notes:
        Spin-orbital labeling convention: :math:`i \equiv (k, \sigma) = 2 k + \sigma`
        with :math:`i`: site index and :math:`\sigma`: spin index


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


def make_tot_density_op(n_sites):
    """Construct total density operator

    .. math::
        N = \sum_{i \sigma} n_{i\sigma}

    Returns:
        ElectronicStructureHamiltonian: the total density operator N
    """
    nqbit = 2 * n_sites
    tot_density = np.zeros((nqbit, nqbit))
    for i in range(n_sites):
        for sig in [0, 1]:
            tot_density[2 * i + sig, 2 * i + sig] = 1

    tot_density_op = ElectronicStructureHamiltonian(hpq=tot_density)

    return tot_density_op

class SpinHamiltonian(Hamiltonian):
    """Ensures restrocompatibility of old SpinHamiltonian class with new Hamiltonian class"""

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("The SpinHamiltonian class is deprecated. Please use the Hamiltonian class instead.", stacklevel=2)
        return super().__init__(*args, **kwargs)

class FermionHamiltonian(Hamiltonian):
    """Ensures restrocompatibility of old SpinHamiltonian class with new Hamiltonian class"""

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("The FermionHamiltonian class is deprecated. Please use the Hamiltonian class instead.", stacklevel=2)
        return super().__init__(*args, **kwargs)
