# -*- coding: utf-8 -*-
"""
Fermion-spin transforms
"""

import itertools
from math import floor
from typing import Tuple, List, Union
from bitstring import BitArray
from anytree import Node
import numpy as np

from qat.core import Term

from .util import tobin
from .hamiltonians import SpinHamiltonian, FermionHamiltonian, ElectronicStructureHamiltonian


def make_fenwick_tree(n):
    ftree = {n - 1: Node(str(n - 1))}

    def fenwick(left, right):
        if left != right:
            lr_half = floor((left + right) / 2.0)
            ftree[lr_half] = Node(str(lr_half), parent=ftree[right])
            fenwick(left, lr_half)
            fenwick(lr_half + 1, right)
        else:
            return None

    fenwick(0, n - 1)
    return [ftree[k] for k in sorted(ftree.keys())]


def _C_set(j, ftree_nodes):
    """Set of children with indices less than j
    of all ancestors of j"""
    res = set()
    for j_anc in ftree_nodes[j].ancestors:
        for kid in j_anc.children:
            if int(kid.name) < j:
                res.add(int(kid.name))
    return res


def _U_set(j, ftree_nodes):
    """Set of all ancestors of j"""
    res = set()
    for j_anc in ftree_nodes[j].ancestors:
        res.add(int(j_anc.name))
    return res


def _F_set(j, ftree_nodes):
    """Children of j"""
    res = set()
    for kid in ftree_nodes[j].children:
        res.add(int(kid.name))
    return res


def _P_set(j, ftree_nodes):
    """P = C u F"""
    return _C_set(j, ftree_nodes).union(_F_set(j, ftree_nodes))


def make_PCU_sets(nqbits: int) -> List[Tuple]:
    """
    Args:
        nqbits (int): Total number of qubits.

    Returns:
        list(set, set, set): The three sets P, C, and U.
    """
    set_list = []
    ftree_nodes = make_fenwick_tree(nqbits)
    for qb in range(nqbits):
        set_list.append((_P_set(qb, ftree_nodes), _C_set(qb, ftree_nodes), _U_set(qb, ftree_nodes)))
    return set_list


def transform_to_jw_basis(fermion_hamiltonian: Union[FermionHamiltonian, ElectronicStructureHamiltonian]) -> SpinHamiltonian:
    r"""Transform to Jordan-Wigner (JW) basis.

    Args:
        fermion_hamiltonian (Union[FermionHamiltonian, ElectronicStructureHamiltonian]): The fermionic hamiltonian.

    Returns:
        SpinHamiltonian: Hamiltonian in spin representation.

    Examples:

    .. run-block:: python

        from qat.core import Term
        from qat.fermion import FermionHamiltonian
        from qat.fermion.transforms import transform_to_jw_basis

        hamiltonian = FermionHamiltonian(
            2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])

        spin_hamiltonian = transform_to_jw_basis(hamiltonian)

        print(f"H = {hamiltonian} \n")
        print(f"H(spin) = {spin_hamiltonian}")

    """

    nqbits = fermion_hamiltonian.nbqbits
    spin_hamiltonian = SpinHamiltonian(nqbits, [], constant_coeff=fermion_hamiltonian.constant_coeff)

    for term in fermion_hamiltonian.terms:

        cur_ham = SpinHamiltonian(nqbits, [], constant_coeff=term.coeff)

        for op, qb in zip(term.op, term.qbits):

            mini_ham = SpinHamiltonian(nqbits, [])
            qbits = list(range(qb + 1))

            st = "Z" * (qb) + "X"
            mini_ham.add_term(Term(0.5, st, qbits))

            st = "Z" * (qb) + "Y"
            sign = -1 if op == "C" else 1
            mini_ham.add_term(Term(1j * sign * 0.5, st, qbits))

            cur_ham = cur_ham * mini_ham

        spin_hamiltonian += cur_ham

    return spin_hamiltonian


def transform_to_parity_basis(fermion_hamiltonian: Union[FermionHamiltonian, ElectronicStructureHamiltonian]) -> SpinHamiltonian:
    r"""Transform to parity basis.

    Args:
        fermion_hamiltonian (Union[FermionHamiltonian, ElectronicStructureHamiltonian]): The fermionic hamiltonian.

    Returns:
        SpinHamiltonian: Hamiltonian in parity spin representation.

    Examples:

    .. run-block:: python

        from qat.core import Term
        from qat.fermion import FermionHamiltonian
        from qat.fermion.transforms import transform_to_parity_basis

        hamiltonian = FermionHamiltonian(
            2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])

        spin_hamiltonian = transform_to_parity_basis(hamiltonian)

        print(f"H = {hamiltonian} \n")
        print(f"H(spin) = {spin_hamiltonian}")

    """

    nqbits = fermion_hamiltonian.nbqbits
    spin_hamiltonian = SpinHamiltonian(nqbits, [], constant_coeff=fermion_hamiltonian.constant_coeff)

    for term in fermion_hamiltonian.terms:

        cur_ham = SpinHamiltonian(nqbits, [Term(term.coeff, "I" * nqbits, list(range(nqbits)))])

        for op, qb in zip(term.op, term.qbits):

            sign = -1 if op == "C" else 1
            mini_ham = SpinHamiltonian(nqbits, [])
            qbits = list(range(qb - 1 if qb > 0 else qb, nqbits))
            st = ("Z" if qb > 0 else "") + "X" + "X" * (nqbits - qb - 1)
            mini_ham.add_term(Term(0.5, st, qbits))

            qbits = list(range(qb, nqbits))
            st = "Y" + "X" * (nqbits - qb - 1)
            mini_ham.add_term(Term(1j * sign * 0.5, st, qbits))

            cur_ham = cur_ham * mini_ham

        spin_hamiltonian += cur_ham

    return spin_hamiltonian


def transform_to_bk_basis(fermion_hamiltonian: Union[FermionHamiltonian, ElectronicStructureHamiltonian]) -> SpinHamiltonian:
    r"""Transform to Bravyi-Kitaev (BK) basis.

    Args:
        fermion_hamiltonian (Union[FermionHamiltonian, ElectronicStructureHamiltonian]): The fermionic hamiltonian.

    Returns:
        SpinHamiltonian: Hamiltonian in BK spin representation.

    Examples:

    .. run-block:: python

        from qat.core import Term
        from qat.fermion import FermionHamiltonian
        from qat.fermion.transforms import transform_to_bk_basis

        hamiltonian = FermionHamiltonian(
            2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])

        spin_hamiltonian = transform_to_bk_basis(hamiltonian)

        print(f"H = {hamiltonian} \n")
        print(f"H(spin) = {spin_hamiltonian}")

    """

    nqbits = fermion_hamiltonian.nbqbits
    pcu_sets = make_PCU_sets(nqbits)

    spin_hamiltonian = SpinHamiltonian(nqbits, [], constant_coeff=fermion_hamiltonian.constant_coeff)

    for term in fermion_hamiltonian.terms:

        cur_ham = SpinHamiltonian(
            nqbits,
            [Term(term.coeff, "I" * nqbits, list(range(nqbits)))],
        )

        for op, qb in zip(term.op, term.qbits):

            sign = -1 if op == "C" else 1
            mini_ham = SpinHamiltonian(nqbits, [])
            p_set, c_set, u_set = pcu_sets[qb]

            qbits = []

            st = "Z" * len(p_set)
            qbits.extend([ind for ind in p_set])
            st += "X"
            qbits.append(qb)
            st += "X" * len(u_set)
            qbits.extend([ind for ind in u_set])
            mini_ham.add_term(Term(0.5, st, qbits))

            qbits = []

            st = "Z" * len(c_set)
            qbits.extend([ind for ind in c_set])
            st += "Y"
            qbits.append(qb)
            st += "X" * len(u_set)
            qbits.extend([ind for ind in u_set])
            mini_ham.add_term(Term(1j * sign * 0.5, st, qbits))

            cur_ham = cur_ham * mini_ham

        spin_hamiltonian += cur_ham

    return spin_hamiltonian


def get_jw_code(nbits: int) -> np.ndarray:
    r"""Construct Jordan-Wigner code matrix :math:`C`.

    i.e matrix :math:`C` to get new bit value :math:`p_i` from bit values :math:`f_j`
    in occupation number basis:

    .. math::
            p_i = \sum_{j} C_{ji} f_j


    Args:
        nqbits (int): Total number of qubits.

    Returns:
        np.ndarray: The C matrix. This is the identity because JW also uses occupation number basis.

    """

    return np.identity(nbits, dtype=int)


def get_parity_code(nbits: int) -> np.ndarray:
    r"""Construct parity code matrix :math:`C`.

    i.e matrix :math:`C` to get new bit value :math:`p_i` from bit values :math:`f_j` in occupation number basis:

    .. math::
            p_i = \sum_{j} C_{ji} f_j


    Args:
        nqbits (int): Total number of qubits.

    Returns:
        np.array: The C matrix.

    """

    c_mat = np.zeros((nbits, nbits), dtype=int)

    for i, j in itertools.product(range(nbits), repeat=2):

        if i <= j:
            c_mat[i, j] = 1

    return c_mat


def get_bk_code(nqbits: int) -> np.ndarray:
    r"""Construct Bravyi-Kitaev code matrix :math:`C`.

    i.e matrix :math:`C` to get new bit value :math:`p_i` from bit values :math:`f_j` in occupation number basis:

    .. math::
            p_i = \sum_{j} C_{ji} f_j

    Args:
        nqbits (int): Total number of qubits.

    Returns:
        np.array: The C matrix.

    """

    ftree_nodes = make_fenwick_tree(nqbits)

    # compute which bits contribute to new bit value
    sublists = {}

    for j in range(nqbits):

        kids = [int(j_anc.name) for j_anc in ftree_nodes[j].children]
        sublists[j] = [j]

        for kid in kids:

            for grandkid in sublists[kid]:
                sublists[j].append(grandkid)

    bk_mat = np.zeros((nqbits, nqbits), dtype=int)

    for i in range(nqbits):
        for j in sublists[i]:

            bk_mat[j, i] = 1

    return bk_mat


def recode_integer(integer: int, code: np.ndarray) -> int:
    r"""Transform integer to other integer

    The bit to bit transform is defined as:

    .. math::
        p_i = \sum_{j} C_{ji} f_j \mathrm{mod.} 2

    Args:
        integer (int): The integer (with binary repr. :math:`|f_0, f_1, \dots f_{n-1}\rangle`) to be converted to new
        representation.
        code (np.ndarray): C matrix to convert bits from one representation to the other.

    Returns:
        int: The integer with binary representation :math:`|p_0, p_1, \dots, p_{n-1}\rangle`.

    """

    nbits = code.shape[0]
    bitstring = tobin(integer, nbits)

    bitarray = [int(x) for x in bitstring]

    res_bitarray = [str(int(c)) for c in code.T.dot(bitarray) % 2]
    res_bitstring = "0b" + "".join(res_bitarray)

    bitarr = BitArray(res_bitstring)

    res_int = bitarr.uint

    return res_int


def change_encoding(mat: np.ndarray, code: np.ndarray) -> np.ndarray:
    r"""
    Change encoding of a matrix A:

    ..math ::
        B[C[i], C[j]] = A[i, j]

    Args:
        mat (np.ndarray): The 2^n x 2^n matrix A to be encoded.
        code (np.ndarray): The nxn code matrix C.

    Returns:
        np.ndarray: The encoded matrix B.
    """

    corresp_table = np.array([0 for _ in range(2 ** code.shape[0])], dtype=int)

    for i in range(2 ** code.shape[0]):
        corresp_table[i] = recode_integer(i, code)

    res_mat = np.zeros(mat.shape, mat.dtype)

    for i, j in itertools.product(range(2 ** code.shape[0]), repeat=2):
        res_mat[corresp_table[i], corresp_table[j]] = mat[i, j]

    return res_mat
