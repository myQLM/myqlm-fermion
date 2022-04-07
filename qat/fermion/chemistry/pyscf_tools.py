#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools for pySCF interfacing

"""
from functools import reduce
import numpy as np
from typing import Union, List

from pyscf import gto, scf, fci, mp, ci
from pyscf import ao2mo


def compute_integrals(molecule: Union[np.ndarray, gto.Mole], mo_coeff, hcore):
    """
    For a given molecule, compute 1-body and 2-body integrals


    Args:
        molecule (): _description_
        mo_coeff (_type_): _description_
        hcore (_type_): _description_

    Returns:
        _type_: _description_
    """
    # no spin dof
    one_electron_compressed = reduce(np.dot, (mo_coeff.T, hcore, mo_coeff))
    n_orbs = mo_coeff.shape[1]
    one_electron_integrals = one_electron_compressed.reshape(n_orbs, n_orbs).astype(
        float
    )
    two_electron_compressed = ao2mo.kernel(molecule, mo_coeff)
    # 1: no permutation symmetry
    two_electron_integrals = ao2mo.restore(1, two_electron_compressed, n_orbs)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order="C"
    )
    return one_electron_integrals, two_electron_integrals


def perform_pyscf_computation(
    geometry, basis, spin, charge, verbose=False, run_FCI=True
):
    # define molecule in pySCF format
    molecule = gto.Mole()
    molecule.atom = geometry
    molecule.basis = basis
    molecule.spin = spin
    molecule.charge = charge
    molecule.symmetry = False
    molecule.build()

    # Run SCF.
    scf_worker = scf.ROHF(molecule) if molecule.spin else scf.RHF(molecule)
    scf_worker.verbose = 0
    scf_worker.run()
    hf_energy = float(scf_worker.e_tot)

    if verbose:
        print("HF energy=", hf_energy)

    one_body_integrals, two_body_integrals = compute_integrals(
        molecule, scf_worker.mo_coeff, scf_worker.get_hcore()
    )

    # overlap_integrals = pyscf_scf.get_ovlp()
    # n_orbitals = int(molecule.nao_nr())
    # n_qubits = 2 * molecule.n_orbitals
    # canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    nuclear_repulsion = float(molecule.energy_nuc())
    orbital_energies = scf_worker.mo_energy.astype(float)
    nels = molecule.nelectron

    # Run CISD
    cisd = ci.CISD(scf_worker)
    cisd.verbose = 0
    cisd.run()
    rdm1 = cisd.make_rdm1()

    # Run MP2.
    # note: molecule.spin must be 0
    mp2 = mp.MP2(scf_worker)
    mp2.verbose = 0
    mp2.run()
    mp2_energy = scf_worker.e_tot + mp2.e_corr

    if verbose:
        print("MP2 energy=", mp2_energy)

    # Run FCI
    if run_FCI:
        fci_worker = fci.FCI(molecule, scf_worker.mo_coeff)
        fci_worker.verbose = 0
        fci_energy = fci_worker.kernel()[0]
    else:
        fci_energy = None
    if verbose:
        print("FCI energy=", fci_energy)

    info = {"MP2": mp2_energy, "FCI": fci_energy, "HF": hf_energy}

    return (
        rdm1,
        orbital_energies,
        nuclear_repulsion,
        nels,
        one_body_integrals,
        two_body_integrals,
        info,
    )
