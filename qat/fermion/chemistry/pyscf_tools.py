# -*- coding: utf-8 -*-
"""
Tools for pySCF interfacing
"""

from functools import reduce
from typing import Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import pyscf


def compute_integrals(molecule: Union[np.ndarray, "pyscf.gto.Mole"], mo_coeff, hcore):
    """
    For a given molecule, compute 1-body and 2-body integrals.
    """

    try:
        # pylint: disable=import-outside-toplevel
        from pyscf import ao2mo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PySCF modules could not be found. Please make sure you installed the PySCF package.") from exc

    # no spin dof
    one_electron_compressed = reduce(np.dot, (mo_coeff.T, hcore, mo_coeff))
    n_orbs = mo_coeff.shape[1]
    one_electron_integrals = one_electron_compressed.reshape(n_orbs, n_orbs).astype(float)
    two_electron_compressed = ao2mo.kernel(molecule, mo_coeff)
    # 1: no permutation symmetry
    two_electron_integrals = ao2mo.restore(1, two_electron_compressed, n_orbs)
    two_electron_integrals = np.asarray(two_electron_integrals.transpose(0, 2, 3, 1), order="C")
    return one_electron_integrals, two_electron_integrals


def perform_pyscf_computation(geometry: list, basis: str, spin: int, charge: int, run_fci: bool = False):
    r"""Perform various calculations using PySCF. This function is a helper function meant to kickstart molecule studies. Its use is
    completely optional, and using other methods or packages is entirely possible.

    This function will compute:

       * The reduced density matrix,
       * The orbital energies,
       * The nuclear repulsion constant,
       * The number of electrons,
       * The one- and two-body integrals,
       * The groundstate energies obtained through Hartree-Fock and 2nd order Möller-Plesset perturbation approach,
       * (Optional) The groundstate energy using the full configuration interaction (full CI) approach.

    Note:
        - The FCI computation is very expensive for big molecules. Enable it only for small molecules !

    Args:
        geometry (list): Defines the molecular structure. The internal format is PySCF format:

            .. code-block::

                atom = [[atom1, (x, y, z)],
                        [atom2, (x, y, z)],
                        ...
                        [atomN, (x, y, z)]]

        basis (str): Defines the basis set.
        spin (int): 2S, number of alpha electrons - number beta electrons to control multiplicity. If spin is None, multiplicity
        will be guessed based on the neutral molecule.
        charge (int): Charge of molecule. Affects the electron numbers.
        run_fci (bool, optional): Whether the groundstates energies should also be computed using a full CI approach. Defaults to
            False.

    Returns:

        Tuple[np.ndarray, list, float, int, np.ndarray, np.ndarray, dict]:
            - rdm1 (np.ndarray): Reduced density matrix.
            - orbital_energies (list): List of orbital energies.
            - nuclear_repulsion (float): Nuclear repulsion constant.
            - nels (int): Number of electrons.
            - one_body_integrals (np.ndarray): One-body integral.
            - two_body_integrals (np.ndarray): Two-body integral.
            - info (dict): Dictionary containing the Hartree-Fock and 2nd order Möller-Plesset computed ground state energies (and optionally the Full CI energy if run_fci is set to True).

    """
    try:
        # pylint: disable=import-outside-toplevel
        from pyscf import gto, scf, fci, mp, ci
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PySCF modules could not be found. Please make sure you installed the PySCF package.") from exc

    # Define molecule in pySCF format
    molecule = gto.Mole()
    molecule.atom = geometry
    molecule.basis = basis
    molecule.spin = spin
    molecule.charge = charge
    molecule.symmetry = False
    molecule.build()

    # Run SCF
    scf_worker = scf.ROHF(molecule) if molecule.spin else scf.RHF(molecule)
    scf_worker.verbose = 0
    scf_worker.run()
    hf_energy = float(scf_worker.e_tot)

    one_body_integrals, two_body_integrals = compute_integrals(molecule, scf_worker.mo_coeff, scf_worker.get_hcore())
    nuclear_repulsion = float(molecule.energy_nuc())
    orbital_energies = scf_worker.mo_energy.astype(float)
    nels = molecule.nelectron

    # Run CISD
    cisd = ci.CISD(scf_worker)
    cisd.verbose = 0
    cisd.run()
    rdm1 = cisd.make_rdm1()

    # Run MP2
    # note: molecule.spin must be 0
    mp2 = mp.MP2(scf_worker)
    mp2.verbose = 0
    mp2.run()
    mp2_energy = scf_worker.e_tot + mp2.e_corr

    # Run FCI
    if run_fci:
        fci_worker = fci.FCI(molecule, scf_worker.mo_coeff)
        fci_worker.verbose = 0
        fci_energy = fci_worker.kernel()[0]
    else:
        fci_energy = None

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
