# -*- coding: utf-8 -*-
"""
Unitary tests for Hamiltonians
"""

import pytest
import numpy as np

from qat.core import Observable, Term

from qat.fermion.hamiltonians import SpinHamiltonian, FermionHamiltonian


def test_spin_hamiltonian():
    obs = Observable(
        2,
        pauli_terms=[
            Term(0.5, "X", [0]),
            Term(0.25, "Y", [1]),
            Term(1.0, "ZZ", [0, 1]),
        ],
    )

    # convert to Hamiltonian
    spin_hamiltonian = SpinHamiltonian(obs.nbqbits, obs.terms, obs.constant_coeff)

    # get matrix
    spin_h_matrix = spin_hamiltonian.get_matrix()

    # diagonalize
    eigvals, eigvecs = np.linalg.eigh(spin_h_matrix)

    print("Eigenvalues=", eigvals)
    np.testing.assert_almost_equal(min(eigvals), -1.25, decimal=13)


def test_to_spin_mapping():

    hamiltonian = FermionHamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CcCc", [0, 1, 1, 0])])
    h_matrix = hamiltonian.get_matrix()

    spin_h = hamiltonian.to_spin()
    spin_matrix = spin_h.get_matrix()

    for term in spin_h.terms:
        assert "C" not in term.op
        assert "c" not in term.op

    np.testing.assert_equal(h_matrix, spin_matrix)


def test_to_electronic_conversion():

    compatible_hamiltonian = FermionHamiltonian(2, [Term(0.3, "Cc", [0, 1]), Term(1.4, "CCcc", [0, 1, 1, 0])])

    elec_hamiltonian = compatible_hamiltonian.to_electronic()

    np.testing.assert_equal(compatible_hamiltonian.get_matrix(), elec_hamiltonian.get_matrix())


def test_to_elec_normal_ordering():

    h = FermionHamiltonian(2, terms=[Term(1, "Cc", [0, 0]), Term(1, "CcCc", [0, 1, 1, 0])])
    h_elec = h.to_electronic()
    print(h_elec)
    assert len(h.terms) == 2
    for t in h_elec.terms:
        if t.op == "Cc":
            assert t.coeff == 2
            assert t.qbits == [0, 0]
        if t.op == "CCcc":
            assert t.coeff == 1
            assert t.qbits == [0, 1, 0, 1]


def test_spin_hamiltonian_compatiiblity_check():

    with pytest.raises(TypeError):
        _ = SpinHamiltonian(2, terms=[Term(1.0, "Cc", [0, 1])])


def test_fermion_hamiltonian_compatiiblity_check():

    with pytest.raises(TypeError):
        _ = FermionHamiltonian(2, terms=[Term(1.0, "XY", [0, 1])])


def test_equivalence_hamiltonian_spectrum():

    nqbits = 3

    H_fermion = FermionHamiltonian(nqbits, [Term(1.0, "Cc", [0, 1]), Term(0.5, "CCcc", [0, 1, 0, 1])])
    H_spin = H_fermion.to_spin()
    H_elec = H_fermion.to_electronic()

    fermion_spectrum = sorted(np.linalg.eigvals(H_fermion.get_matrix()))
    spin_spectrum = sorted(np.linalg.eigvals(H_spin.get_matrix()))
    electronic_spectrum = sorted(np.linalg.eigvals(H_elec.get_matrix()))

    np.testing.assert_equal(fermion_spectrum, spin_spectrum)
    np.testing.assert_equal(fermion_spectrum, electronic_spectrum)


def test_hamiltonian_addition_constant():
    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])
    assert h1.constant_coeff == 0

    h = h1 + 2
    assert h.constant_coeff == 2.0


def test_hamiltonian_addition_constant_reversed():
    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])
    assert h1.constant_coeff == 0

    h = 2 + h1
    assert h.constant_coeff == 2.0


def test_hamiltonian_addition():
    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])
    h2 = FermionHamiltonian(2, terms=[Term(1.0, "CCcc", [0, 1, 1, 0])])

    h = h1 + h2

    term = h.terms[0]
    assert (term.coeff, term.op, term.qbits) == (2j, "CCcc", [0, 1, 0, 1])


def test_hamiltonian_subtraction_constant():
    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])
    assert h1.constant_coeff == 0

    h = h1 - 2
    assert h.constant_coeff == -2.0


def test_hamiltonian_subtraction_constant_reversed():
    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])
    assert h1.constant_coeff == 0

    h = 2 - h1
    assert h.constant_coeff == 2.0

    term = h.terms[0]
    assert (term.coeff, term.op, term.qbits) == ((-1 - 2j), "CCcc", [0, 1, 0, 1])


def test_hamiltonian_subtraction():
    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])
    h2 = FermionHamiltonian(2, terms=[Term(1.0, "CCcc", [0, 1, 1, 0])])

    h = h1 - h2

    term = h.terms[0]
    assert (term.coeff, term.op, term.qbits) == ((2 + 2j), "CCcc", [0, 1, 0, 1])


def test_hamiltonian_multiplication_constant():

    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])

    h = h1 * 2

    term = h.terms[0]
    assert (term.coeff, term.op, term.qbits) == ((2 + 4j), "CCcc", [0, 1, 0, 1])


def test_hamiltonian_multiplication_constant_reversed():

    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])

    h = 2 * h1

    term = h.terms[0]
    assert (term.coeff, term.op, term.qbits) == ((2 + 4j), "CCcc", [0, 1, 0, 1])


def test_hamiltonian_multiplication():

    h1 = FermionHamiltonian(2, terms=[Term(1.0 + 2 * 1j, "CCcc", [1, 0, 1, 0])])
    h2 = FermionHamiltonian(2, terms=[Term(1.0, "CCcc", [0, 1, 1, 0])])

    h = h1 * h2

    term = h.terms[0]
    assert (term.coeff, term.op, term.qbits, h.constant_coeff) == ((1 + 2j), "CCcc", [0, 1, 0, 1], 0.0)


def test_get_matrix_changes_spin():

    obs = Observable(
        2,
        pauli_terms=[
            Term(0.5, "X", [0]),
            Term(0.25, "Y", [1]),
            Term(1.0, "ZZ", [0, 1]),
        ],
    )

    # convert to Hamiltonian
    spin_hamiltonian = SpinHamiltonian(obs.nbqbits, obs.terms, obs.constant_coeff)

    matrix1 = spin_hamiltonian.get_matrix()

    # Set new terms
    spin_hamiltonian.set_terms([
        Term(0.5, "Z", [0]),
        Term(0.5, "X", [1]),
        Term(1.0, "XX", [0, 1])
    ])

    matrix2 = spin_hamiltonian.get_matrix()

    # Check array are different
    assert not np.all(np.equal(matrix1, matrix2))

def test_get_matrix_changes_fermion():

    H_fermion = FermionHamiltonian(2, [Term(1.0, "Cc", [0, 1]), Term(0.5, "CCcc", [0, 1, 0, 1])])

    matrix1 = H_fermion.get_matrix()

    H_fermion.set_terms([Term(0.2, "Cc", [0, 1]), Term(2, "CCcc", [0, 1, 1, 0])])

    matrix2 = H_fermion.get_matrix()

    # Check array are different
    assert not np.all(np.equal(matrix1, matrix2))
    
def test_get_matrix_change_electronic():
    
    H_fermion = FermionHamiltonian(2, [Term(1.0, "Cc", [0, 1]), Term(0.5, "CCcc", [0, 1, 0, 1])])
    H_elec = H_fermion.to_electronic()

    matrix1 = H_elec.get_matrix()

    H_elec.hpqrs = np.zeros(H_elec.hpqrs.shape)

    matrix2 = H_elec.get_matrix()

    # Check array are different
    assert not np.all(np.equal(matrix1, matrix2))

    H_elec.constant_coeff += 3
    matrix3 = H_elec.get_matrix()

    assert np.all(np.equal(matrix3, matrix2 + 3*np.eye(4,4)))



def test_get_matrix_sparse_spin():

    obs = Observable(
        2,
        pauli_terms=[
            Term(0.5, "X", [0]),
            Term(0.25, "Y", [1]),
            Term(1.0, "ZZ", [0, 1]),
        ],
    )

    # convert to Hamiltonian
    spin_hamiltonian = SpinHamiltonian(obs.nbqbits, obs.terms, obs.constant_coeff)

    expected = np.array(
        [
            [1.0 + 0.0j, 0.0 - 0.25j, 0.5 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.25j, -1.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
            [0.5 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, 0.0 - 0.25j],
            [0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.25j, 1.0 + 0.0j],
        ]
    )

    np.testing.assert_equal(spin_hamiltonian.get_matrix(), expected)
    np.testing.assert_equal(spin_hamiltonian.get_matrix(sparse=True).toarray(), expected)


def test_get_matrix_sparse_fermion():

    H_fermion = FermionHamiltonian(2, [Term(1.0, "Cc", [0, 1]), Term(0.5, "CCcc", [0, 1, 0, 1])])

    expected = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.5 + 0.0j],
        ]
    )

    np.testing.assert_equal(H_fermion.get_matrix(), expected)
    np.testing.assert_equal(H_fermion.get_matrix(sparse=True).toarray(), expected)


def test_get_matrix_sparse_electronic():

    H_fermion = FermionHamiltonian(2, [Term(1.0, "Cc", [0, 1]), Term(0.5, "CCcc", [0, 1, 0, 1])])
    H_elec = H_fermion.to_electronic()

    expected = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.5 + 0.0j],
        ]
    )

    np.testing.assert_equal(H_elec.get_matrix(), expected)
    np.testing.assert_equal(H_elec.get_matrix(sparse=True).toarray(), expected)

def test_electronic_hamiltonian_dag():
    H_fermion = FermionHamiltonian(2, [Term(1.0j, "Cc", [0, 1]), Term(0.5j, "CCcc", [0, 1, 0, 1])])
    H_elec = H_fermion.to_electronic()
    
    np.testing.assert_equal(np.conj(H_elec.get_matrix()), H_elec.dag().get_matrix())
