#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fermion-spin transforms
"""

import itertools
from math import floor
from bitstring import BitArray
from anytree import Node
import numpy as np
from qat.core import Term
from qat.fermion.util import tobin
from qat.fermion.hamiltonians import Hamiltonian


def make_fenwick_tree(N):
    ftree = {N - 1: Node(str(N - 1))}

    def fenwick(left, right):
        if left != right:
            lr_half = floor((left + right) / 2.0)
            ftree[lr_half] = Node(str(lr_half), parent=ftree[right])
            fenwick(left, lr_half)
            fenwick(lr_half + 1, right)
        else:
            return

    fenwick(0, N - 1)
    return [ftree[k] for k in sorted(ftree.keys())]


def _C_set(j, ftree_nodes):
    """set of children with indices less than j
    of all ancestors of j"""
    res = set()
    for j_anc in ftree_nodes[j].ancestors:
        for kid in j_anc.children:
            if int(kid.name) < j:
                res.add(int(kid.name))
    return res


def _U_set(j, ftree_nodes):
    """set of all ancestors of j"""
    res = set()
    for j_anc in ftree_nodes[j].ancestors:
        res.add(int(j_anc.name))
    return res


def _F_set(j, ftree_nodes):
    """children of j"""
    res = set()
    for kid in ftree_nodes[j].children:
        res.add(int(kid.name))
    return res


def _P_set(j, ftree_nodes):
    """P = C u F"""
    return _C_set(j, ftree_nodes).union(_F_set(j, ftree_nodes))


# N : number of qbits


def make_PCU_sets(nqbits):
    """
    Args:
        qb (int): qubit index
        nqbits (int): total number of qubits

    Returns:
        list(set, set, set): the three sets P, C, and U
    """
    set_list = []
    ftree_nodes = make_fenwick_tree(nqbits)
    for qb in range(nqbits):
        set_list.append(
            (_P_set(qb, ftree_nodes), _C_set(qb, ftree_nodes), _U_set(qb, ftree_nodes))
        )
    return set_list


# basis transforms
def transform_to_jw_basis(fermion_hamiltonian):
    """Transform to Jordan-Wigner (JW) basis

    Args:
        fermion_hamiltonian (Hamiltonian or ElectronicStructureHamiltonian): the
            fermionic hamiltonian

    Returns:
        Hamiltonian: the same hamiltonian, in JW spin representation

    Examples:

    .. run-block:: python

        from qat.core import Term
        from qat.fermion import Hamiltonian

        hamiltonian = Hamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
        print("H = ", hamiltonian)

        from qat.fermion.transforms import transform_to_jw_basis
        spin_hamiltonian = transform_to_jw_basis(hamiltonian)
        print("H(spin) = ", spin_hamiltonian)

    """

    nqbits = fermion_hamiltonian.nbqbits
    spin_hamiltonian = Hamiltonian(
        nqbits, [], constant_coeff=fermion_hamiltonian.constant_coeff, do_clean_up=False
    )
    for term in fermion_hamiltonian.terms:
        cur_ham = Hamiltonian(nqbits, [], constant_coeff=term.coeff)
        for op, qb in zip(term.op, term.qbits):
            mini_ham = Hamiltonian(nqbits, [])
            qbits = list(range(qb + 1))

            st = "Z" * (qb) + "X"
            mini_ham.add_term(Term(0.5, st, qbits))

            st = "Z" * (qb) + "Y"
            sign = -1 if op == "C" else 1
            mini_ham.add_term(Term(1j * sign * 0.5, st, qbits))

            cur_ham = cur_ham * mini_ham
        spin_hamiltonian += cur_ham
    spin_hamiltonian.clean_up()
    return spin_hamiltonian


def transform_to_parity_basis(fermion_hamiltonian):
    """Transform to parity basis

    Args:
        fermion_hamiltonian (Hamiltonian or ElectronicStructureHamiltonian): the
            fermionic hamiltonian

    Returns:
        Hamiltonian: the same hamiltonian, in parity spin representation

    Examples:

    .. run-block:: python

        from qat.core import Term
        from qat.fermion import Hamiltonian

        hamiltonian = Hamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
        print("H = ", hamiltonian)

        from qat.fermion.transforms import transform_to_parity_basis
        spin_hamiltonian = transform_to_parity_basis(hamiltonian)
        print("H(spin) = ", spin_hamiltonian)
    """
    nqbits = fermion_hamiltonian.nbqbits
    spin_hamiltonian = Hamiltonian(
        nqbits, [], constant_coeff=fermion_hamiltonian.constant_coeff, do_clean_up=False
    )
    for term in fermion_hamiltonian.terms:
        cur_ham = Hamiltonian(
            nqbits, [Term(term.coeff, "I" * nqbits, list(range(nqbits)))]
        )
        for op, qb in zip(term.op, term.qbits):
            sign = -1 if op == "C" else 1
            mini_ham = Hamiltonian(nqbits, [])
            qbits = list(range(qb - 1 if qb > 0 else qb, nqbits))
            st = ("Z" if qb > 0 else "") + "X" + "X" * (nqbits - qb - 1)
            mini_ham.add_term(Term(0.5, st, qbits))

            qbits = list(range(qb, nqbits))
            st = "Y" + "X" * (nqbits - qb - 1)
            mini_ham.add_term(Term(1j * sign * 0.5, st, qbits))

            cur_ham = cur_ham * mini_ham
        spin_hamiltonian += cur_ham
    spin_hamiltonian.clean_up()
    return spin_hamiltonian


def transform_to_bk_basis(fermion_hamiltonian):
    """Transform to Bravyi-Kitaev (BK) basis

    Args:
        fermion_hamiltonian (Hamiltonian or ElectronicStructureHamiltonian): the
            fermionic hamiltonian

    Returns:
        Hamiltonian: the same hamiltonian, in BK spin representation

    Examples:

    .. run-block:: python

        from qat.core import Term
        from qat.fermion import Hamiltonian

        hamiltonian = Hamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
        print("H = ", hamiltonian)

        from qat.fermion.transforms import transform_to_bk_basis
        spin_hamiltonian = transform_to_bk_basis(hamiltonian)
        print("H(spin) = ", spin_hamiltonian)
    """
    nqbits = fermion_hamiltonian.nbqbits
    pcu_sets = make_PCU_sets(nqbits)
    spin_hamiltonian = Hamiltonian(
        nqbits, [], constant_coeff=fermion_hamiltonian.constant_coeff, do_clean_up=False
    )
    for term in fermion_hamiltonian.terms:
        cur_ham = Hamiltonian(
            nqbits,
            [Term(term.coeff, "I" * nqbits, list(range(nqbits)))],
            do_clean_up=False,
        )
        for op, qb in zip(term.op, term.qbits):
            sign = -1 if op == "C" else 1
            mini_ham = Hamiltonian(nqbits, [], do_clean_up=False)
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
    spin_hamiltonian.clean_up()
    return spin_hamiltonian


#  codes  #
def get_jw_code(nbits):
    """Construct Jordan-Wigner code matrix :math:`C`

    i.e matrix :math:`C` to get new bit value :math:`p_i` from bit values :math:`f_j`
    in occupation number basis:

    .. math::
            p_i = \sum_{j} C_{ji} f_j


    Args:
        nqbits (int): total number of qubits

    Returns:
        np.array: the C matrix (here, the identity because JW also uses occupation
        number basis)
    """
    return np.identity(nbits, dtype=int)


def get_parity_code(nbits):
    """Construct parity code matrix :math:`C`

    i.e matrix :math:`C` to get new bit value :math:`p_i` from bit values :math:`f_j`
    in occupation number basis:

    .. math::
            p_i = \sum_{j} C_{ji} f_j


    Args:
        nqbits (int): total number of qubits

    Returns:
        np.array: the C matrix
    """
    c_mat = np.zeros((nbits, nbits), dtype=int)
    for i, j in itertools.product(range(nbits), repeat=2):
        if i <= j:
            c_mat[i, j] = 1
    return c_mat


def get_bk_code(nqbits):
    """Construct Bravyi-Kitaev code matrix :math:`C`

    i.e matrix :math:`C` to get new bit value :math:`p_i` from bit values :math:`f_j`
    in occupation number basis:

    .. math::
            p_i = \sum_{j} C_{ji} f_j

    Args:
        nqbits (int): total number of qubits

    Returns:
        np.array: the C matrix
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


def recode_integer(integer, code):
    r"""Transform integer to other integer

    The bit to bit transform is defined as:

    .. math::
        p_i = \sum_{j} C_{ji} f_j \mathrm{mod.} 2

    Args:
        integer (int): the integer
            (with binary repr. :math:`|f_0, f_1, \dots f_{n-1}\rangle`)
            to be converted to new representation
        code (np.array): C matrix to convert bits from one representation
            to the other.

    Returns:
        int: the integer with binary representation :math:`|p_0, p_1, \dots, p_{n-1}\rangle`.
    """
    nbits = code.shape[0]
    bitstring = tobin(integer, nbits)
    bitarray = [int(x) for x in bitstring]
    res_bitarray = [str(int(c)) for c in code.T.dot(bitarray) % 2]
    res_bitstring = "0b" + "".join(res_bitarray)
    bitarr = BitArray(res_bitstring)
    res_int = bitarr.uint
    return res_int


def change_encoding(mat, code):
    """
    Change encoding of a matrix A:

    ..math ::
        B[C[i], C[j]] = A[i, j]

    Args:
        mat (np.array): the 2^n x 2^n matrix A to be encoded
        code (np.array): the nxn code matrix C

    Returns:
        np.array: the encoded matrix B
    """
    corresp_table = np.array([0 for _ in range(2 ** code.shape[0])], dtype=int)
    for i in range(2 ** code.shape[0]):
        corresp_table[i] = recode_integer(i, code)

    res_mat = np.zeros(mat.shape, mat.dtype)
    for i, j in itertools.product(range(2 ** code.shape[0]), repeat=2):
        res_mat[corresp_table[i], corresp_table[j]] = mat[i, j]

    return res_mat
