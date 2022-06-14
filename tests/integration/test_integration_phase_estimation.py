#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from itertools import product
from qat.lang.AQASM import H, X, QRoutine, Program
from qat.lang.AQASM.gates import AbstractGate
from qat.fermion.hamiltonians import ElectronicStructureHamiltonian, make_hubbard_model
from qat.fermion.transforms import transform_to_jw_basis
from qat.fermion.phase_estimation import (
    perform_phase_estimation,
    apply_adiabatic_state_prep,
)
from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation
from qat.fermion.chemistry.ucc import convert_to_h_integrals
from qat.qpus import get_default_qpu


def make_hubbard_dimer(U, t_hopping):
    nqbit = 4

    # set h_pq
    # by convention: (i, sig) = 2*i + sig with i: site index and sig: spin index
    hpq = np.zeros((nqbit, nqbit))
    # hopping
    for sig in [0, 1]:
        hpq[2 * 0 + sig, 2 * 1 + sig] = -t_hopping
        hpq[2 * 1 + sig, 2 * 0 + sig] = -t_hopping
    # chemical potential
    for i in [0, 1]:
        for sig in [0, 1]:
            hpq[2 * i + sig, 2 * i + sig] = -U / 2

    # h_pqrs: Hubbard interaction
    hpqrs = np.zeros((nqbit, nqbit, nqbit, nqbit))
    for i in [0, 1]:
        for sig in [0, 1]:
            hpqrs[2 * i + sig, 2 * i + 1 - sig, 2 * i + sig, 2 * i + 1 - sig] = -U

    return ElectronicStructureHamiltonian(hpq=hpq, hpqrs=hpqrs)


def check_all_states(
    H_el_structure,
    nqbits_phase,
    n_trotter_steps,
    n_adiab_steps,
    E_target,
    E_range,
    delta,
    second_E_range=0.1,
    user_second_try=False,
):
    """
    Test that whatever the initial state, it always finds an energy from the expected eigenvalues
    The quality of this test largly depends on nqbits_phase, n_trotter_steps, E_target, E_range
    However, once an energy is found, even if it's quite off, a real user will adjust the parameters so
    that the energy is more precise. We test for this via running check_all_states() with user_second_try=True.
    """

    # Calculate the corresponding eigenvalues and eigenvectors
    # H_spin = transform_to_jw_basis(H_el_structure)
    H_spin_mat = H_el_structure.get_matrix()
    eigvals, eigvecs_transposed = np.linalg.eigh(H_spin_mat)
    print(eigvals)

    n_qubits = H_el_structure.nbqbits
    count_off_results = 0
    for init_vec in product(range(2), repeat=n_qubits):
        init_vec = "".join(str(init_qubit) for init_qubit in init_vec)
        qpe_energy, probs = perform_phase_estimation(
            H_el_structure,
            nqbits_phase,
            n_trotter_steps,
            init_vec=init_vec,
            n_adiab_steps=n_adiab_steps,
            E_target=E_target,
            size_interval=E_range,
        )
        # Check if the energy found is among the eigenenergies up to a delta
        energy_is_found = False
        for i, eigenenergy in enumerate(eigvals):
            if abs(qpe_energy - eigenenergy) <= delta:
                print(qpe_energy, eigenenergy)
                energy_is_found = True
                break

        # Simulate the user's behaviour of trying to find a more precise energy with E_target="last found energy"
        if not energy_is_found and user_second_try:
            qpe_energy, probs = perform_phase_estimation(
                H_el_structure,
                nqbits_phase,
                n_trotter_steps,
                init_vec=init_vec,
                n_adiab_steps=n_adiab_steps,
                E_target=qpe_energy,
                size_interval=second_E_range,
            )
            for i, eigenenergy in enumerate(eigvals):
                if abs(qpe_energy - eigenenergy) <= delta:
                    energy_is_found = True
                    print(eigenenergy, qpe_energy)
                    break
        if not energy_is_found:
            print(qpe_energy, "not found")
            count_off_results += 1
    return count_off_results


def check_basic(t_hopping):
    """
    For a custom made Hamiltonian, check that solving with QPE returns an energy
    close to the expected eigenenergies.
    """
    # Specify the Hamiltonian
    U = 1.62
    H_hubbard = make_hubbard_dimer(U, t_hopping)

    H_el_structure_2 = ElectronicStructureHamiltonian(H_hubbard.hpq * 2.18, H_hubbard.hpqrs * 3.27, -4.2)
    # Define the QPE parameters
    nqbits_phase = 4  # 4
    n_trotter_steps = 7  # 7
    delta = 10 * 1.0 / 2 ** (nqbits_phase + 1)

    n_adiab_steps = 0  # 0
    E_target = -4  # -4
    E_range = 15  # 15

    assert (
        check_all_states(
            H_el_structure_2,
            nqbits_phase,
            n_trotter_steps,
            n_adiab_steps,
            E_target,
            E_range,
            delta,
        )
        <= 2
    )


# Create an H2 molecule
geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
basis = "sto-3g"
spin = 0  # = 2 S with S total spin angular momentum = # of unpaired electrons
charge = 0

# Generate the problem
(
    rdm1,
    orbital_energies,
    nuclear_repulsion,
    nels,
    one_body_integrals,
    two_body_integrals,
    info,
) = perform_pyscf_computation(geometry=geometry, basis=basis, spin=spin, charge=charge)
# Create the Electronic Structure Hamiltonian
hpq, hpqrs = convert_to_h_integrals(one_body_integrals, two_body_integrals)
H_el_structure = ElectronicStructureHamiltonian(hpq, hpqrs, nuclear_repulsion)

# Modify the Hamiltonian to have a wider range eigenvalues, far from 0
hpq_off = -3.01
hpqrs_factor = 3.64
constant_E = -14.8
H_el_structure_modified = ElectronicStructureHamiltonian(hpq + hpq_off, hpqrs * hpqrs_factor, nuclear_repulsion + constant_E)
lowest_E = -27.003  # this is the result from diagonalising the above H


def test_adiabatic_state_prep():
    """
    If there is an adiabatic state preparation check that it will collapse
    every possible initial state to an eigenvector of the final Hamiltonian.
    """
    H_qbasis = transform_to_jw_basis(H_el_structure)
    n_qubits_H = H_qbasis.nbqbits
    # set it to 2 just for the QPEs to execute
    n_phase_bits = 2

    # "_hopping" is for the part of the H which has only hpp terms (no hpq or hpqrs)
    # extract the diagonal and return an empty array but with this diagonal.
    H_el_hopping_hpq = np.diag(np.diag(H_el_structure.hpq))
    # "_f" stands for fermionic basis
    H_el_hopping_f = ElectronicStructureHamiltonian(H_el_hopping_hpq, hpqrs=None, constant_coeff=0.0)
    H_el_hopping_qbasis = transform_to_jw_basis(H_el_hopping_f)

    # Rarely more than one of the 16 states is not returned (second for-loop).
    # In this case, just do the computation again.
    # If it fails again, then there is most probably a problem.
    n_right_states = 0
    computations_are_right = False
    for _ in range(2):

        # Try all the computational basis state vectors as init_vec
        for init_vec_i, init_vec in enumerate(product(range(2), repeat=n_qubits_H)):
            init_vec = "".join(str(init_qubit) for init_qubit in init_vec)
            print(init_vec)

            # Initialize the program
            prog = Program()
            phase_reg = prog.qalloc(n_phase_bits)
            data_reg = prog.qalloc(n_qubits_H)

            # Apply an X gate to every qubit needing to be in state '1'.
            for i, state in enumerate(init_vec):
                if state == "1":
                    prog.apply(X, data_reg[i])

            # Adiabatic state preparation using intermediate QPE circuits - in an outer function
            # so as to be separately tested
            n_adiab_steps = 5
            n_trotter_steps = 5
            apply_adiabatic_state_prep(
                prog,
                n_adiab_steps,
                H_el_hopping_qbasis,
                H_qbasis,
                n_trotter_steps,
                phase_reg,
                data_reg,
            )

            circ = prog.to_circ()
            res = get_default_qpu().submit(circ.to_job(qubits=data_reg, nbshots=0))
            for sample in res.raw_data:
                if sample.state.int == init_vec_i and sample.probability > 0.949:
                    n_right_states += 1
            print(sample.state, sample.state.int, sample.probability)
        print(n_right_states)

        # If it got the right computations, no need for further checking
        if n_right_states >= 15:
            computations_are_right = True
            break
        else:
            n_right_states = 0

    assert computations_are_right


@pytest.mark.skip(reason="Phase estimation initialized with QRoutine not working with PyLinalg for now.")
def test_hubbard_molecule__from_notebook():
    """
    A test for the Hubbard molecule which we present in the qpe_hubbard_molecule notebook
    in qat-tutorial. We check if the QPE can find every eigenenergy of the H for the respective
    eigenvector, i.e. checking the core functionality of the QPE (upon specifying a good enough
    energy window, i.e. enclosing the eigenenergies it solves for).
    Every eigenvector is specified through STATE_PREPARATION, i.e. via giving a QRoutine with
    this AbstractGate to "init_vec".
    """

    # Create Hamiltonian
    U = 4.57
    t = 1.16
    t_mat = -t * np.array([[0.0, 1.0], [1.0, 0.0]])
    hamilt = make_hubbard_model(t_mat, U, mu=U / 2)  # U / 2

    # Extract eigenvectors and eigenvalues to test for
    H_mat = hamilt.get_matrix()
    eigvals, eigvecs_transposed = np.linalg.eigh(H_mat)
    eigvecs = eigvecs_transposed.T
    np.set_printoptions(precision=4, suppress=True)
    print(eigvals, end="\n\n")

    # Set QPE parameters
    nqbits_phase = 8
    n_trotter_steps = 8
    n_adiab_steps = 0
    E_target = 0
    size_interval = 16
    delta = 10 * 1.0 / 2 ** (nqbits_phase + 1)

    count_energies_off = 0
    for state_index in range(len(eigvals)):  # one eigenstate per eigenenergy
        eigen_state = eigvecs[state_index, :]

        qrout = QRoutine()
        qrout.new_wires(4)
        state_prep = AbstractGate("STATE_PREPARATION", [np.ndarray])
        state_normalizing_factor = np.sqrt(sum(coeff**2 for coeff in eigen_state))
        qrout.apply(
            state_prep(np.array(eigen_state) / state_normalizing_factor),
            [0, 1, 2, 3],
        )

        qpe_energy, _ = perform_phase_estimation(
            hamilt,
            nqbits_phase,
            n_trotter_steps,
            n_adiab_steps=n_adiab_steps,
            E_target=E_target,
            size_interval=size_interval,
            init_vec=qrout,
        )

        if abs(eigvals[state_index] - qpe_energy) > delta:
            count_energies_off += 1
    assert count_energies_off <= 2


def test_rough_estimate():
    """
    Test if a rough first try (i.e. nqbits_phase and n_trotter_steps are just 6 and 5) gives close
    enough energies to the expected ones.
    """

    # Define the QPE parameters
    nqbits_phase = 6  # 6
    n_trotter_steps = 5  # 5
    delta = 10 * 1.0 / 2 ** (nqbits_phase + 1)

    n_adiab_steps = 0
    E_target = 0
    E_range = 3

    assert (
        check_all_states(
            H_el_structure,
            nqbits_phase,
            n_trotter_steps,
            n_adiab_steps,
            E_target,
            E_range,
            delta,
        )
        <= 1
    )


def test_precise_estimate():
    """
    Test if aiming for more precise energies works if one uses the already found energy as E_target.
    """
    # Define the QPE parameters
    nqbits_phase = 9
    n_trotter_steps = 11
    delta = 1 / 2**nqbits_phase

    n_adiab_steps = 0
    E_target = 0
    E_range = 3

    assert (
        check_all_states(
            H_el_structure,
            nqbits_phase,
            n_trotter_steps,
            n_adiab_steps,
            E_target,
            E_range,
            delta,
            user_second_try=True,
        )
        <= 1
    )


def test_bad_E_target_E_range():
    """
    Test that poor initial E_target and E_range give poor results.
    """
    # Define the QPE parameters
    nqbits_phase = 9
    n_trotter_steps = 9
    delta = 10 * 1.0 / 2 ** (nqbits_phase + 1)

    n_adiab_steps = 10
    E_target = -5
    E_range = 20

    assert (
        check_all_states(
            H_el_structure_modified,
            nqbits_phase,
            n_trotter_steps,
            n_adiab_steps,
            E_target,
            E_range,
            delta,
        )
        >= 10
    )


def test_E_target_E_range_adiab_state_prep():
    """
    Test that a Hamiltonian stemming from the H2 one, but with larger eigenvalue range and far from 0
    is still well handled.
    """
    # Define the QPE parameters
    nqbits_phase = 4
    n_trotter_steps = 5
    delta = 10 * 1.0 / 2 ** (nqbits_phase + 1)

    n_adiab_steps = 0
    E_target = -20
    E_range = 25
    second_E_range = 4

    assert (
        check_all_states(
            H_el_structure_modified,
            nqbits_phase,
            n_trotter_steps,
            n_adiab_steps,
            E_target,
            E_range,
            delta,
            second_E_range,
            user_second_try=True,
        )
        <= 2
    )


def test_no_init_vec():
    """
    Show that one can supply a routine preparing a desired initial state and it works.
    """
    # Define the QPE parameters
    nqbits_phase = 4
    n_trotter_steps = 5
    delta = 10 * 1.0 / 2 ** (nqbits_phase + 1)

    n_adiab_steps = 0
    E_target = constant_E
    E_range = 40

    # Perform the QPE
    qpe_energy, probs = perform_phase_estimation(
        H_el_structure_modified,
        nqbits_phase,
        n_trotter_steps,
        init_vec=None,
        n_adiab_steps=n_adiab_steps,
        E_target=E_target,
        size_interval=E_range,
    )

    # Calculate the corresponding eigenvalues and eigenvectors
    H_spin_mat = H_el_structure_modified.get_matrix()
    eigvals, eigvecs_transposed = np.linalg.eigh(H_spin_mat)

    # Check if the energy found is among the eigenenergies up to a delta
    energy_is_found = False
    for i, eigenenergy in enumerate(eigvals):
        if abs(qpe_energy - eigenenergy) <= delta:
            energy_is_found = True
            break
    assert energy_is_found


def test_init_vec_as_routine():
    """
    Show that one can supply a routine preparing a desired initial state and it works.
    """
    # Define the QPE parameters
    nqbits_phase = 4
    n_trotter_steps = 5
    delta = 10 * 1.0 / 2 ** (nqbits_phase + 1)

    n_adiab_steps = 0
    E_target = constant_E
    E_range = 40

    # The QRoutine preparing the initial vector
    n_qubits_H = H_el_structure_modified.nbqbits
    q_routine_init_state = QRoutine()
    data_reg = q_routine_init_state.new_wires(n_qubits_H)
    for i in range(n_qubits_H):
        q_routine_init_state.apply(H, data_reg[i])
    init_vec = q_routine_init_state

    # Perform the QPE
    qpe_energy, probs = perform_phase_estimation(
        H_el_structure_modified,
        nqbits_phase,
        n_trotter_steps,
        init_vec=init_vec,
        n_adiab_steps=n_adiab_steps,
        E_target=E_target,
        size_interval=E_range,
    )
    assert abs(lowest_E - qpe_energy) <= delta


def test_basic_zero_hopping():
    check_basic(t_hopping=0)


def test_basic_finite_hopping():
    check_basic(t_hopping=0.05)


def test_basic_finite_hopping_2():
    check_basic(t_hopping=0.1)


def test_basic_finite_hopping_3():
    check_basic(t_hopping=0.25)


def test_basic_finite_hopping_4():
    check_basic(t_hopping=0.5)


def test_basic_finite_hopping_5():
    check_basic(t_hopping=1.0)


def test_basic_finite_hopping_6():
    check_basic(t_hopping=1.3)


def test_basic_finite_hopping_7():
    check_basic(t_hopping=1.6)
