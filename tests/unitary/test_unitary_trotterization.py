import numpy as np
import scipy as sp
from qat.core import Observable, Term

from qat.fermion.util import get_unitary_from_circuit
from qat.fermion.trotterisation import (
    _excitation_operator_jw,
    _coulomb_exchange_operator_jw,
)
from qat.fermion.trotterisation import (
    _number_excitation_operator_jw,
    _double_excitation_operator_jw,
)
from qat.fermion.trotterisation import _number_operator_jw, make_trotter_slice_jw, make_spin_hamiltonian_trotter_slice
from qat.fermion.hamiltonians import Hamiltonian, ElectronicStructureHamiltonian

TOL1 = 4
TOL2 = 10


def check_trotterisation(hpq, hpqrs, tol):
    hamiltonian = ElectronicStructureHamiltonian(hpq, hpqrs)

    U_exact = sp.linalg.expm(-1j * hamiltonian.get_matrix())
    U_trotter = get_unitary_from_circuit(make_trotter_slice_jw(hpq, hpqrs, 1), hpq.shape[0])

    np.testing.assert_almost_equal(np.linalg.norm(U_exact - U_trotter), 0, decimal=tol)


def test_make_trotter_slice_jw_2_qbits():
    nqbits = 2
    hpq = np.zeros((nqbits, nqbits))
    hpqrs = np.zeros((nqbits, nqbits, nqbits, nqbits))
    hpq[0][0] = 1
    check_trotterisation(hpq, hpqrs, tol=1e-10)


def test_make_trotter_slice_jw_2_qbits_2():
    nqbits = 2
    hpq = np.zeros((nqbits, nqbits))
    hpqrs = np.zeros((nqbits, nqbits, nqbits, nqbits))
    hpq[0][0] = 1
    hpqrs[0][1][1][0] = hpqrs[1][0][0][1] = 1
    check_trotterisation(hpq, hpqrs, tol=1e-10)


def test_make_trotter_slice_jw_2_qbits_3():
    nqbits = 2
    hpq = np.zeros((nqbits, nqbits))
    hpqrs = np.zeros((nqbits, nqbits, nqbits, nqbits))
    hpq[0][1] = 1
    hpq[1][0] = 1
    check_trotterisation(hpq, hpqrs, tol=1e-10)


def test_make_trotter_slice_jw_4_qbits():
    nqbits = 4
    hpq = np.zeros((nqbits, nqbits))
    hpqrs = np.zeros((nqbits, nqbits, nqbits, nqbits))
    hpqrs[3][2][1][0] = hpqrs[0][1][2][3] = 1
    check_trotterisation(hpq, hpqrs, tol=1e-10)


def checks_trotter_slice(hpq, hpqrs):
    number_qubits = hpq.shape[0]
    hamiltonian = ElectronicStructureHamiltonian(hpq, hpqrs)
    a = sp.linalg.expm(-1j * hamiltonian.get_matrix())
    rout1 = make_trotter_slice_jw(hpq, hpqrs, 1)
    b = get_unitary_from_circuit(rout1, number_qubits)
    assert np.linalg.norm(a - b) < TOL1

    return a


def test_hpq_number_operator():
    number_qubits = 2
    hpq = np.zeros((2, 2))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpq[1][1] = 2

    c = get_unitary_from_circuit(_number_operator_jw(hpq, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1


def test_hpq_excitation_operator():
    number_qubits = 2
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpq[0][1] = 2
    hpq[1][0] = 2

    c = get_unitary_from_circuit(_excitation_operator_jw(hpq, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1

    number_qubits = 5
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpq[4][0] = 2
    hpq[0][4] = 2

    c = get_unitary_from_circuit(_excitation_operator_jw(hpq, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1


def test_hpqqp_coulomb_operator():
    number_qubits = 3
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[0][2][2][0] = -1.5

    c = get_unitary_from_circuit(_coulomb_exchange_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1

    number_qubits = 4
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[3][0][0][3] = 6.5489

    c = get_unitary_from_circuit(_coulomb_exchange_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1


def test_hpqqr_numberexcitation_operator():
    # cas r<q<p
    number_qubits = 3
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[2][1][1][0] = -1.5
    hpqrs[0][1][1][2] = -1.5
    c = get_unitary_from_circuit(_number_excitation_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1

    # cas q<r
    number_qubits = 5
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[4][0][0][1] = -1.5
    hpqrs[1][0][0][4] = -1.5
    c = get_unitary_from_circuit(_number_excitation_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1

    # Cas q>p
    number_qubits = 5
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[2][4][4][0] = -1.5
    hpqrs[0][4][4][2] = -1.5
    c = get_unitary_from_circuit(_number_excitation_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1


def test_hpqrs_doubleexcitation_operator():
    # cas 4 qubits test de base pour verifier le circuit le plus simple (sans gestion du circuit optimise)
    number_qubits = 4
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[3][2][1][0] = -2
    hpqrs[0][1][2][3] = -2

    c = get_unitary_from_circuit(_double_excitation_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1

    # cas 6 qubits prise en compte circuit optimise
    number_qubits = 6
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[5][3][2][0] = 2.25
    hpqrs[0][2][3][5] = 2.25

    c = get_unitary_from_circuit(_double_excitation_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1

    # Second cas gestion porte Z controlee eloignee et de l'ecartement entre q et r
    number_qubits = 6
    hpq = np.zeros((number_qubits, number_qubits))
    hpqrs = np.zeros((number_qubits, number_qubits, number_qubits, number_qubits))
    hpqrs[5][3][1][0] = 4.587
    hpqrs[0][1][3][5] = 4.587

    c = get_unitary_from_circuit(_double_excitation_operator_jw(hpqrs / 2, 1), number_qubits)
    a = checks_trotter_slice(hpq, hpqrs)
    assert np.linalg.norm(c - a) < TOL1


def check_evolution(ising_hamiltonian):
    hmat = ising_hamiltonian.get_matrix()
    a = sp.linalg.expm(-1j * hmat)
    nqbit = ising_hamiltonian.nbqbits
    Qrout = make_spin_hamiltonian_trotter_slice(ising_hamiltonian)
    b = get_unitary_from_circuit(Qrout, nqbit)
    np.testing.assert_almost_equal(np.linalg.norm(a - b), 0, decimal=3)


def test_ZZ_term():
    theta = 0.543
    spin_hamiltonian = Hamiltonian(2, terms=[Term(theta, "ZZ", [0, 1])])
    check_evolution(spin_hamiltonian)


def test_XX_term():
    theta = 0.3
    spin_hamiltonian = Hamiltonian(2, terms=[Term(theta, "XX", [0, 1])])
    check_evolution(spin_hamiltonian)


def test_n_terms():
    nqbits = 2
    n_terms = 10
    paulis = ["X", "Y", "Z"]
    pauli_dict = {}
    pauli_dict["Z"] = np.array([[1, 0], [0, -1]])
    pauli_dict["Y"] = np.array([[0, -1j], [1j, 0]])
    pauli_dict["X"] = np.array([[0, 1], [1, 0]])
    pauli_dict["I"] = np.array([[1, 0], [0, 1]])

    terms = ["".join(list(np.random.choice(paulis, 2))) for _ in range(n_terms)]
    terms = list(set(terms))
    n_terms = len(terms)
    coeffs = np.random.randn(n_terms)
    U_mats = [sp.linalg.expm(-1j * coeff * np.kron(pauli_dict[key[0]], pauli_dict[key[1]])) for coeff, key in zip(coeffs, terms)]
    U_mat = np.linalg.multi_dot(U_mats[::-1])
    terms = [Term(coeff, term, [0, 1]) for coeff, term in zip(coeffs, terms)]
    qrout2 = make_spin_hamiltonian_trotter_slice(Observable(2, pauli_terms=terms))
    U_mat_circ = get_unitary_from_circuit(qrout2, nqbits)
    # print(terms)
    np.testing.assert_almost_equal(np.linalg.norm(U_mat - U_mat_circ), 0, decimal=13)
