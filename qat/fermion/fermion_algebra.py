# -*- coding: utf-8 -*-
"""
Fermionic algebra tools
"""

from numbers import Number
from copy import deepcopy
from typing import List, Optional, Union

from qat.core import Term
from qat.core.variables import BaseArithmetic


PAULI_MATS = {
    "X": [[0, 1], [1, 0]],
    "I": [[1, 0], [0, 1]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
}


class FermionicTerm(Term):
    """
    Implementation of the FermionicTerm class. This class is mostly used for overloading operations, which allows operations
    between terms containing fermionic operators.
    """

    def __init__(self, coefficient: complex, op: str, qbits: List[int], do_validity_check: bool = True):

        if any(value in PAULI_MATS for value in op):
            raise TypeError("FermionicTerm only accepts fermionic operators C and c.")

        super(FermionicTerm, self).__init__(coefficient, op, qbits, do_validity_check)

    def __mul__(self, other):

        if isinstance(other, (Number, BaseArithmetic)):
            new_term = self.copy()
            new_term.coeff *= other
            return new_term

        term = self.copy()
        term.op += other.op
        term.qbits += other.qbits
        term.coeff *= other.coeff

        return term

    def copy(self):
        """Deepcopy the current class.

        Returns:
            :class:`~qat.fermion.hamiltonians.FermionicTerm`: Copy of the FermionicTerm.
        """
        return deepcopy(self)

    @staticmethod
    def from_term(term: Term) -> "FermionicTerm":
        """Converts a Term class to a FermionicTerm class.

        Args:
            term (Term): Term.

        Returns:
            FermionicTerm
        """

        return FermionicTerm(term.coeff, term.op, term.qbits)


def permute_fermionic_operator(fermionic_term, ind) -> List[Term]:
    """
    Perform the permutation of the two operators in index ind and ind + 1 in a fermionic Term pauli string

    Args:
        fermionic_term (Term): the fermionic term which operators we seek to permute
        ind (int): the lower index of the two consecutive creation or annihilation operators we seek to permute

    Returns:
        list_terms (List[Term]): the list of fermionic terms resulting of the permutation
    """

    coeff = fermionic_term.coeff
    pauli_op = fermionic_term.op
    qbits = fermionic_term.qbits

    if ind >= len(pauli_op) - 1:
        raise IndexError

    permuted_pauli_op = pauli_op[:ind] + pauli_op[ind + 1] + pauli_op[ind] + pauli_op[ind + 2 :]
    permuted_qbits = qbits[:]
    permuted_qbits[ind], permuted_qbits[ind + 1] = permuted_qbits[ind + 1], permuted_qbits[ind]

    if "c" in pauli_op[ind : ind + 2] and "C" in pauli_op[ind : ind + 2] and qbits[ind] == qbits[ind + 1]:

        return [
            FermionicTerm(coefficient=coeff, op=pauli_op[:ind] + pauli_op[ind + 2 :], qbits=qbits[:ind] + qbits[ind + 2 :]),
            FermionicTerm(coefficient=-coeff, op=permuted_pauli_op, qbits=permuted_qbits),
        ]

    else:
        return [FermionicTerm(coefficient=-coeff, op=permuted_pauli_op, qbits=permuted_qbits)]


def order_qubits(fermionic_term) -> Term:
    """
    Takes a fermionic term whose pauli_op is supposed to be normal-ordered, and reorder it increasing qbit numbers

    Args:
        fermionic_term (Term): the term to reorder (it is already normal-ordered)

    Returns:
        ordered_term (Term): the reordered term
    """

    coeff = fermionic_term.coeff
    pauli_op = fermionic_term.op
    qbits = fermionic_term.qbits

    if "c" in pauli_op:

        ind_c = pauli_op.index("c")
        qbits_C = qbits[:ind_c]
        qbits_c = qbits[ind_c:]

    else:
        ind_C = pauli_op.index("C")
        qbits_C = qbits[:ind_C]
        qbits_c = qbits[ind_C:]

    new_qbits = []
    for qbits_op in [qbits_C, qbits_c]:

        qbits_temp = qbits_op[:]
        ordered = False

        while not ordered:

            ind = 0
            while ind < len(qbits_temp) - 1 and qbits_temp[ind] <= qbits_temp[ind + 1]:

                if qbits_temp[ind] == qbits_temp[ind + 1]:
                    return
                ind += 1

            if ind < len(qbits_temp) - 1:

                ind += 1
                new_ind = 0

                while qbits_temp[new_ind] < qbits_temp[ind]:
                    new_ind += 1

                elt_not_in_order = qbits_temp.pop(ind)
                qbits_temp.insert(new_ind, elt_not_in_order)
                coeff *= (-1) ** (ind - new_ind)

            else:
                ordered = True

        new_qbits += qbits_temp

    return FermionicTerm(coefficient=coeff, op=pauli_op, qbits=new_qbits)


def normal_order_fermionic_ops(fermionic_term) -> List[Term]:
    """
    Order the operators list of a fermionic_term by putting the creations operators
    on the left and the annihilation operators on the right, with respect to the fermionic anticommutation relations.

    Args:
        fermionic_term (Term): the term to order

    Returns:
        ordered_fermionic_terms (List[Term]): The list of ordered fermionic terms
    """

    pauli_op = fermionic_term.op

    # No operator ordering needed if term contains only one type of femionic operator
    if "c" not in pauli_op or "C" not in pauli_op:
        return [fermionic_term]

    # Sanity check
    ind_c = pauli_op.index("c")

    try:
        ind_C = pauli_op[ind_c:].index("C") + ind_c

    except ValueError:
        new_terms = [fermionic_term]

    else:
        new_terms = []
        for new_fermionic_term in permute_fermionic_operator(fermionic_term, ind_C - 1):
            new_terms += normal_order_fermionic_term(new_fermionic_term)

    return new_terms


def are_term_ops_ordered(term):

    op = term.op

    # Check length of fermionic op
    if len(op) <= 1:
        return True, None

    # Check ops are normally ordered
    if "C" in op:
        ind_c_dag = op.index("C")
        if "c" in op[:ind_c_dag]:
            return False, op[:ind_c_dag].index("c") + ind_c_dag - 1

    if "c" in op:
        ind_c = op.index("c")
        if "C" in op[ind_c:]:
            return False, ind_c

    # Check indices are normally ordered
    for idx in range(len(op) - 1):
        if op[idx] == op[idx + 1]:
            if term.qbits[idx] > term.qbits[idx + 1]:
                return False, idx

    return True, None


def normal_order_fermionic_term(fermionic_term) -> List[Term]:
    """
    Order any fermionic term by putting the creation operators on the left,
    ordered by increasing qubit numbers, and the annihilation operators on the right,
    ordered by increasing qubit numbers, with respect to the fermionic anticommutation relations.

    Args:
        fermionic_term (Term): the term to order

    Returns:
        ordered_fermionic_terms (List[Term]): the list of ordered fermionic terms
    """

    new_terms = normal_order_fermionic_ops(fermionic_term)
    ordered_terms = []

    for new_term in new_terms:
        ordered_term = order_qubits(new_term)

        if ordered_term:
            ordered_terms.append(ordered_term)

    return ordered_terms
