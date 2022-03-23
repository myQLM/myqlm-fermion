from typing import Union, List, Iterable
from dataclasses import dataclass

import numpy as np
from copy import deepcopy

from qat.core import Observable, Term
from .ucc import (
    get_cluster_ops,
    transform_integrals_to_new_basis,
    select_active_orbitals,
    compute_active_space_integrals,
    guess_init_state,
    convert_to_h_integrals,
    construct_active_orbitals,
    compute_core_constant,
)
from ..hamiltonians import ElectronicStructureHamiltonian
from collections.abc import Mapping
from typing import Callable


def _copy_doc(copy_func: Callable) -> Callable:
    """
    Copy docstring from one function or class to another.
    """
    # FIXME :  See qat/core/application.py if problems with Sphinx doc
    def wrapper(func: Callable) -> Callable:
        func.__doc__ = copy_func.__doc__
        return func

    return wrapper


class MolecularHamiltonian(Mapping):

    __slots__ = ["one_body_integrals", "two_body_integrals", "nuclear_repulsion"]

    def __init__(
        self,
        one_body_integrals: np.ndarray,
        two_body_integrals: np.ndarray,
        nuclear_repulsion: np.ndarray,
    ):
        """MolecularHamiltonian container class.

        Args:
            one_body_integrals (np.ndarray): One-body integrals
            two_body_integrals (np.ndarray): Two-body integrals
            nuclear_repulsion (np.ndarray): Nuclear repulsion
        """

        self.one_body_integrals = one_body_integrals
        self.two_body_integrals = two_body_integrals
        self.nuclear_repulsion = nuclear_repulsion

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def copy(self):
        return deepcopy(self)

    def transform_basis(self, transformation_matrix: np.ndarray, inplace=False):
        """
        Change one and two body integrals (indices p, q...) to
        new basis (indices i, j...) using transformation U such that

        .. math::
            \hat{c}_{i}=\sum_{q}U_{qi}c_{q}

        i.e

        .. math::

            \hat{I}_{ij} =\sum_{pq}U_{pi}I_{pq}U_{jq}^{\dagger}
            \hat{I}_{ijkl}=\sum_{pqrs}U_{pi}U_{qj}I_{pqrs}U_{kr}^{\dagger}U_{ls}^{\dagger}

        Args:
            transformation_matrix (np.array): transformation matrix :math:`U`

        Returns:
            np.array, np.array: one- and two-body integrals :math:`\hat{I}_{ij}` and :math:`\hat{I}_{ijkl}`
        """
        integrals = transform_integrals_to_new_basis(
            self.one_body_integrals, self.two_body_integrals, transformation_matrix
        )

        if inplace:
            self.one_body_integrals = integrals[0]
            self.two_body_integrals = integrals[1]

        else:
            return MolecularHamiltonian(
                integrals[0], integrals[1], self.nuclear_repulsion
            )


class Molecule(object):
    def __init__(
        self,
        hamiltonian: MolecularHamiltonian,
        n_electrons: int,
        noons: Union[np.ndarray, List[float]],
        orbital_energies: np.ndarray,
    ):
        self.hamiltonian = hamiltonian
        self.n_electrons = n_electrons
        self.noons = noons
        self.orbital_energies = orbital_energies

        self.active_indices = 2*self.hamiltonian.one_body_integrals.shape[0]
        self.occupied_indices = []
        self.core_constant = 0

    @property
    def one_body_integrals(self):
        return self.hamiltonian.one_body_integrals

    @property
    def two_body_integrals(self):
        return self.hamiltonian.two_body_integrals

    @property
    def nuclear_repulsion(self):
        return self.hamiltonian.nuclear_repulsion

    @property
    def n_active_electrons(self):
        if self.occupied_indices is not None:
            return self.n_electrons - 2 * len(self.occupied_indices)

    def copy(self):
        return deepcopy(self)

    @_copy_doc(MolecularHamiltonian.transform_basis)
    def transform_basis(self, transformation_matrix: np.ndarray, inplace=False):
        return self.hamiltonian.transform_basis(transformation_matrix, inplace=inplace)

    def select_active_space(
        self,
        threshold_1: float = 2.0e-2,
        threshold_2: float = 2.0e-3,
        inplace=False,
    ):

        active_indices, occupied_indices = select_active_orbitals(
            noons=self.noons,
            nb_e=self.n_electrons,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
        )

        if inplace:

            self.active_indices, self.occupied_indices = (
                active_indices,
                occupied_indices,
            )

            (
                self.core_constant,
                self.hamiltonian.one_body_integrals,
                self.hamiltonian.two_body_integrals,
            ) = compute_active_space_integrals(
                self.hamiltonian.one_body_integrals,
                self.hamiltonian.two_body_integrals,
                self.active_indices,
                self.occupied_indices,
            )

        else:

            molecule = self.copy()

            molecule.active_indices, molecule.occupied_indices = (
                active_indices,
                occupied_indices,
            )
            (
                molecule.core_constant,
                molecule.hamiltonian.one_body_integrals,
                molecule.hamiltonian.two_body_integrals,
            ) = compute_active_space_integrals(
                molecule.hamiltonian.one_body_integrals,
                molecule.hamiltonian.two_body_integrals,
                active_indices,
                occupied_indices,
            )

            return molecule

    def get_active_orbitals_info(self):
        """Utility function which computes active orbital related informations. 

        Returns:
            n_active_electrons: Number of active electrons
            active_noons: Active natural orbital occupation numbers
            active_orbital_energies: Active oribtal energies
        """
        active_noons, active_orbital_energies = [], []

        for ind in self.active_indices:
            active_noons.extend([self.noons[ind], self.noons[ind]])
            active_orbital_energies.extend(
                [self.orbital_energies[ind], self.orbital_energies[ind]]
            )

        n_active_electrons = self.n_electrons - 2 * len(self.occupied_indices)

        return n_active_electrons, active_noons, active_orbital_energies

    def guess_init_state(self):
        """Find initial guess using Møller-Plesset perturbation theory.

        The trial parametrization is efficiently improved upon the
        Hartree-Fock solution (which would set every initial parameter to
        zero) thanks to the following formula identifying the UCC parameters
        in the Møller-Plesset (MP2) solution :

        .. math::

            \theta_a^i = 0

        .. math::

            \theta_{a, b}^{i, j} = \frac{h_{a, b, i, j} -
            h_{a, b, j, i}}{\epsilon_i + \epsilon_j -\epsilon_a -
            \epsilon_b}

        where :math:`h_{p, q, r, s}` is the 2-electron molecular orbital integral,
        and :math:`\epsilon_i` is the orbital energy.

        Returns:
            _type_: _description_
        """

        (
            n_active_electrons,
            active_noons,
            active_orbital_energies,
        ) = self.get_active_orbitals_info()

        _, hpqrs = convert_to_h_integrals(
            self.one_body_integrals, self.two_body_integrals
        )
        (
            theta_list,
            ket_hf_init,
            actives_occupied_orbitals,
            actives_unoccupied_orbitals,
        ) = guess_init_state(
            n_active_electrons, active_noons, active_orbital_energies, hpqrs
        )

        return (
            theta_list,
            ket_hf_init,
            active_noons,
            actives_occupied_orbitals,
            actives_unoccupied_orbitals,
        )

    def get_cluster_ops(self):

        n_active_electrons, active_noons, _ = self.get_active_orbitals_info()
        (
            actives_occupied_orbitals,
            actives_unoccupied_orbitals,
        ) = construct_active_orbitals(
            n_active_electrons, list(range(len(active_noons)))
        )

        return get_cluster_ops(
            active_noons, actives_occupied_orbitals, actives_unoccupied_orbitals
        )

    def get_cluster_ops_and_init_guess(self):
        (
            theta_list,
            ket_hf_init,
            _,
            _,
            _,
        ) = self.guess_init_state()

        cluster_list = self.get_cluster_ops()

        return cluster_list, ket_hf_init, theta_list

    def get_electronic_hamiltonian(self):
        """Converts the MolecularHamiltonian to an ElectronicStructureHamiltonian.

        Returns:
            :py:class:`~qat.fermion.ElectronicStructureHamiltonian`: Electronic structure hamiltonian
        """

        hpq, hpqrs = convert_to_h_integrals(
            self.one_body_integrals, self.two_body_integrals
        )

        # self.core_constant = compute_core_constant(
        #     self.one_body_integrals,
        #     self.two_body_integrals,
        #     occupied_indices=occupied_indices,
        # )

        H_electronic = ElectronicStructureHamiltonian(
            hpq,
            hpqrs,
            constant_coeff=self.nuclear_repulsion + self.core_constant,
            do_clean_up=False,
        )

        return H_electronic
