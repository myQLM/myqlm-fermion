# -*- coding: utf-8 -*-
"""
Integration test for fermionic Hamiltonians algebraic operations
"""

import os
from pathlib import Path
import numpy as np

from qat.fermion.chemistry import MolecularHamiltonian, MoleculeInfo
from qat.fermion.chemistry.ucc import get_cluster_ops

resources_path = Path(__file__).resolve().parents[1] / "resources"

h2_data_path = resources_path / "h2_data.npz"
lih_data_path = resources_path / "lih_data.npz"

### H2 integration tests
h2_data = np.load(h2_data_path, allow_pickle=True)

rdm1 = h2_data["rdm1"]
orbital_energies = h2_data["orbital_energies"]
nuclear_repulsion = h2_data["nuclear_repulsion"]
n_electrons = h2_data["n_electrons"]
one_body_integrals = h2_data["one_body_integrals"]
two_body_integrals = h2_data["two_body_integrals"]
info = h2_data["info"].tolist()

nqbits = rdm1.shape[0] * 2
mol_h = MolecularHamiltonian(one_body_integrals, two_body_integrals, nuclear_repulsion)

cluster_ops = get_cluster_ops(n_electrons, nqbits=nqbits)

H = mol_h.get_electronic_hamiltonian()
H_sp = H.to_spin()


def test_h2_elec_to_fermion():

    expected = sorted(np.linalg.eigvals(H.get_matrix()))
    test = sorted(np.linalg.eigvals(H.to_fermion().get_matrix()))

    np.testing.assert_almost_equal(expected, test)


def test_h2_fermionic_commutation():

    for c_op in cluster_ops:

        commutator_fermion = (H | c_op).get_matrix()
        commutator_spin1 = (H_sp | c_op.to_spin()).get_matrix()
        commutator_spin2 = (H.to_spin("jordan-wigner") | c_op.to_spin("jordan-wigner")).get_matrix()
        commutator_spin3 = (H.to_spin("bravyi-kitaev") | c_op.to_spin("bravyi-kitaev")).get_matrix()
        commutator_spin4 = (H.to_spin("parity") | c_op.to_spin("parity")).get_matrix()

        for commutator_spin in [commutator_spin1, commutator_spin2, commutator_spin3, commutator_spin4]:
            fermion_eigvals = np.linalg.eigvals(commutator_fermion)
            spin_eigvals = np.linalg.eigvals(commutator_spin)

            np.testing.assert_almost_equal(sorted(np.real(fermion_eigvals)), sorted(np.real(spin_eigvals)))
            np.testing.assert_almost_equal(sorted(np.imag(fermion_eigvals)), sorted(np.imag(spin_eigvals)))


### LiH test
lih_data = np.load(lih_data_path, allow_pickle=True)

rdm1 = lih_data["rdm1"]
orbital_energies = lih_data["orbital_energies"]
nuclear_repulsion = lih_data["nuclear_repulsion"]
n_electrons = lih_data["n_electrons"]
one_body_integrals = lih_data["one_body_integrals"]
two_body_integrals = lih_data["two_body_integrals"]
info = lih_data["info"].tolist()

nqbits = rdm1.shape[0] * 2

mol_h = MolecularHamiltonian(one_body_integrals, two_body_integrals, nuclear_repulsion)

noons, basis_change = np.linalg.eigh(rdm1)
noons = list(reversed(noons))
basis_change = np.flip(basis_change, axis=1)

mol_h_new_basis = mol_h.transform_basis(basis_change)

molecule = MoleculeInfo(
    mol_h_new_basis,
    n_electrons=n_electrons,
    noons=noons,
    orbital_energies=orbital_energies,
)

molecule.restrict_active_space(threshold_1=0.02, threshold_2=0.002)

cluster_ops = get_cluster_ops(molecule.n_electrons, noons=molecule.noons)

H = molecule.hamiltonian.get_electronic_hamiltonian()
H_sp = H.to_spin()


def test_lih_elec_to_fermion():

    expected = sorted(np.linalg.eigvals(H.get_matrix()))
    test = sorted(np.linalg.eigvals(H.to_fermion().get_matrix()))

    np.testing.assert_almost_equal(expected, test)


def test_lih_fermionic_commutations():

    for c_op in cluster_ops:

        commutator_fermion = (H | c_op).get_matrix()
        commutator_spin1 = (H_sp | c_op.to_spin()).get_matrix()
        commutator_spin2 = (H.to_spin("jordan-wigner") | c_op.to_spin("jordan-wigner")).get_matrix()
        commutator_spin3 = (H.to_spin("bravyi-kitaev") | c_op.to_spin("bravyi-kitaev")).get_matrix()
        commutator_spin4 = (H.to_spin("parity") | c_op.to_spin("parity")).get_matrix()

        for commutator_spin in [commutator_spin1, commutator_spin2, commutator_spin3, commutator_spin4]:
            fermion_eigvals = np.linalg.eigvals(commutator_fermion)
            spin_eigvals = np.linalg.eigvals(commutator_spin)

            np.testing.assert_almost_equal(sorted(np.real(fermion_eigvals)), sorted(np.real(spin_eigvals)))
            np.testing.assert_almost_equal(sorted(np.imag(fermion_eigvals)), sorted(np.imag(spin_eigvals)))
