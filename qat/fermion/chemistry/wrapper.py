from typing import Union, List

import numpy as np
from copy import deepcopy

from .ucc import (
    transform_integrals_to_new_basis,
    select_active_orbitals,
    compute_active_space_integrals,
    convert_to_h_integrals,
)
from ..hamiltonians import ElectronicStructureHamiltonian
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


class MolecularHamiltonian(object):
    def __init__(
        self,
        one_body_integrals: np.ndarray,
        two_body_integrals: np.ndarray,
        constant_coeff: np.ndarray,
    ):
        """MolecularHamiltonian container class.

        Args:
            one_body_integrals (np.ndarray): One-body integrals
            two_body_integrals (np.ndarray): Two-body integrals
            constant_coeff (np.ndarray): Constant coefficient
        """

        self.one_body_integrals = one_body_integrals
        self.two_body_integrals = two_body_integrals
        self.constant_coeff = constant_coeff
        self.core_constant = 0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        s = " MolecularHamiltonian(\n"
        s += f" - constant_coeff : {self.constant_coeff}\n"
        s += f" - integrals shape\n"
        s += f"    * one_body_integral : {self.one_body_integrals.shape}\n"
        s += f"    * two_body_integral : {self.two_body_integrals.shape}\n"

        return s

    def transform_basis(self, transformation_matrix: np.ndarray):
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
            flip (bool): If True, the transformation matrix is reversed.

        Returns:
            np.array, np.array: one- and two-body integrals :math:`\hat{I}_{ij}` and :math:`\hat{I}_{ijkl}`
        """

        integrals = transform_integrals_to_new_basis(
            self.one_body_integrals, self.two_body_integrals, transformation_matrix
        )

        return MolecularHamiltonian(integrals[0], integrals[1], self.constant_coeff)

    def select_active_space(
        self,
        noons,
        n_electrons,
        threshold_1: float = 2.0e-2,
        threshold_2: float = 2.0e-3,
    ):
        r"""Selects the right active space and freezes core electrons
        according to their NOONs :math:`n_i`.

        This function is an implementation of the *Complete Active Space*
        (CAS) approach. It divides orbital space into sets of *active* and
        *inactive* orbitals, the occupation number of the latter remaining
        unchanged during the computation.

        The active space indices are defined as:

        .. math::

            \mathcal{A} = \{i, n_i \in [\varepsilon_2, 2 - \varepsilon_1[\} \cup \{i, n_i \geq 2-\varepsilon_1, 2(i+1)\geq N_e \}

        The inactive occupied orbitals are defined as:

        .. math::

            \mathcal{O} = \{i, n_i \geq 2 -\varepsilon_1, 2(i+1) < N_e \}

        The restriction of the one- and two-body integrals (and update of the core energy)
        is then carried out according to:

        .. math::

            \forall u,v \in \mathcal{A},\; I^{(a)}_{uv} = I_{uv} + \sum_{i\in \mathcal{O}} 2 I_{i,u,v,i} - I_{i,u,i,v}

        .. math::

            \forall u,v,w,x \in \mathcal{A}, I^{(a)}_{uvwx} = I_{uvwx}

        .. math::

            E_\mathrm{core}^{(a)} = E_\mathrm{core} + \sum_{i\in\mathcal{O}} I_{ii} + \sum_{ij\in\mathcal{O}} 2 I_{ijji} - I_{ijij}

        Args:
            one_body_integrals (np.array): 2D array of one-body integrals :math:`I_{uv}`
            two_body_integrals (np.array): 4D array of two-body integrals :math:`I_{uvwx}`
            threshold_1 (float, optional): The upper threshold :math:`\varepsilon_1` on
                the NOON of an active orbital. Defaults to 0.02.
            noons (list<float>): the natural-orbital occupation numbers :math:`n_i`, sorted
                in descending order (from high occupations to low occupations)
            nels (int): The number of electrons :math:`N_e`.
            nuclear_repulsion (float): value of the nuclear repulsion energy :math:`E_\mathrm{core}`.
            threshold_2 (float, optional): The lower threshold :math:`\varepsilon_2` on
                the NOON of an active orbital. Defaults to 0.001.

        Returns:
            MolecularHamiltonian, list<int>, list<int>:

            - the molecular Hamiltonian in active space :math:`H^{(a)}`
            - the list of indices corresponding to the active orbitals, :math:`\mathcal{A}`
            - the list of indices corresponding to the occupied orbitals, :math:`\mathcal{O}`
        """

        active_indices, occupied_indices = select_active_orbitals(
            noons=noons,
            nb_e=n_electrons,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
        )

        hamiltonian = self.copy()

        hamiltonian.active_indices, hamiltonian.occupied_indices = (
            active_indices,
            occupied_indices,
        )

        (
            hamiltonian.core_constant,
            hamiltonian.one_body_integrals,
            hamiltonian.two_body_integrals,
        ) = compute_active_space_integrals(
            hamiltonian.one_body_integrals,
            hamiltonian.two_body_integrals,
            active_indices,
            occupied_indices,
        )

        return hamiltonian, active_indices, occupied_indices

    def get_electronic_hamiltonian(self):
        """Converts the MolecularHamiltonian to an ElectronicStructureHamiltonian.

        Returns:
            :py:class:`~qat.fermion.ElectronicStructureHamiltonian`: Electronic structure hamiltonian
        """

        hpq, hpqrs = convert_to_h_integrals(
            self.one_body_integrals, self.two_body_integrals
        )

        H_electronic = ElectronicStructureHamiltonian(
            hpq,
            hpqrs,
            constant_coeff=self.constant_coeff + self.core_constant,
            do_clean_up=False,
        )

        return H_electronic


class MoleculeInfo(object):
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

        self.active_indices = None
        self.occupied_indices = None

    def __repr__(self):

        h_str = self.hamiltonian.__repr__().replace("*", "**").replace("-", "*")

        s = "MoleculeInfo(\n"
        s += " - MolecularHamiltonian(\n"
        for st in h_str.splitlines()[1:]:
            s += f"   {st}\n"

        s += f" - n_electrons = {self.n_electrons}\n"
        s += f" - noons = {self.noons}\n"
        s += f" - orbital energies = {self.orbital_energies}\n"

        if self.active_indices is not None:
            s += " - active space:\n"
            s += f"    * active indices : {self.active_indices}\n"

        if not all(
            cond is None for cond in (self.active_indices, self.occupied_indices)
        ):
            s += f"    * occupied indices : {self.occupied_indices}\n"

        s += ")"
        return s

    @property
    def one_body_integrals(self):
        return self.hamiltonian.one_body_integrals

    @property
    def two_body_integrals(self):
        return self.hamiltonian.two_body_integrals

    @property
    def constant_coeff(self):
        return self.hamiltonian.constant_coeff

    def copy(self):
        return deepcopy(self)

    def _get_active_orbitals_info(self):
        """Utility function which computes active orbital related informations.

        Returns:
            n_active_electrons: Number of active electrons
            active_noons: Active natural orbital occupation numbers
            active_orbital_energies: Active orbital energies
        """
        active_noons, active_orbital_energies = [], []

        for ind in self.hamiltonian.active_indices:
            active_noons.extend([self.noons[ind], self.noons[ind]])
            active_orbital_energies.extend(
                [self.orbital_energies[ind], self.orbital_energies[ind]]
            )

        n_active_electrons = self.n_electrons - 2 * len(self.occupied_indices)

        return n_active_electrons, active_noons, active_orbital_energies

    def restrict_active_space(
        self, threshold_1: float = 2.0e-2, threshold_2: float = 2.0e-3
    ):

        (
            self.hamiltonian,
            self.active_indices,
            self.occupied_indices,
        ) = self.hamiltonian.select_active_space(
            self.noons, self.n_electrons, threshold_1, threshold_2
        )

    def unpack(self):
        """Allow for the unpacking of a selection of MoleculeInfo attributes.

        Returns:
            n_electrons (int) : Number of electrons
            noons (List[np.ndarray]) : Natural orbitals occupation numbers
            orbital_energies (List[np.ndarray]) : Orbital energies
            active_indices (List[int]) : Actives indices
            occupied_indices (List[int]) : Occupied indices
        """

        output = (
            self.n_electrons,
            self.noons,
            self.orbital_energies,
            self.active_indices,
            self.occupied_indices,
        )

        return output
