# -*- coding: utf-8 -*-
"""
Chemistry wrapper classes
"""

from typing import Dict, Optional, Tuple, Union, List
from copy import deepcopy
import numpy as np

from .ucc import (
    transform_integrals_to_new_basis,
    select_active_orbitals,
    compute_active_space_integrals,
    convert_to_h_integrals,
)
from ..hamiltonians import ElectronicStructureHamiltonian


class MolecularHamiltonian(object):
    r"""
    MolecularHamiltonian helper class. It represents the electronic-structure Hamiltonian defined
    using one- and two-body integrals.

    This electronic-structure Hamiltonian is defined by:

    ..  math::

        H=\sum_{uv\sigma}I_{uv}c^{\dagger}_{u\sigma}c_{v\sigma}+\frac{1}{2}\sum_{uvwx}\sum_{\sigma \sigma'} I_{uvwx}c^{\dagger}_{u\sigma}c^{\dagger}_{v\sigma'}c_{w\sigma'}c_{x\sigma}+r\mathbb{I}

    with :math:`r` the core repulsion constant, and with :math:`I_{uv}` and :math:`I_{uvwx}` the one- and two-body integrals defined
    by:

    ..  math::

        I_{uv} = \int dr \phi^{*}_{u}(r)h_{1}[\phi_{v}(r)]

        I_{uvwx} = \int dr dr' \phi^{*}_{u}(r)\phi^{*}_{v}(r')v[\phi_{w}(r)\phi_{x}(r')]

    Here, :math:`\{\phi_{i}(r)\}_{i=0...N-1}` is the single-particle basis, with :math:`N` the size, which depends on the basis
    chosen. :math:`h_{1} = h_{kin} + h_{pot}` is the one-body Hamiltonian, and :math:`v` the Coulomb operator.

    Note:

        This electronic-structure Hamiltonian definition is different than the one used in
        :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian`.

    Args:

        one_body_integrals (np.ndarray): One-body integral :math:`I_{uv}`.
        two_body_integrals (np.ndarray): Two-body integral :math:`I_{uvwx}`.
        constant_coeff (np.ndarray): Constant coefficient :math:`r` (core repulsion).

    Attributes:

        nqbits (int): The total number of qubits.
        one_body_integrals (np.ndarray): One-body integral :math:`I_{uv}`.
        two_body_integrals (np.ndarray): Two-body integral :math:`I_{uvwx}`.
        constant_coeff (np.ndarray): Constant coefficient :math:`r` (core repulsion).

    Example:

        .. run-block:: python

            import numpy as np
            from qat.fermion.chemistry import MolecularHamiltonian

            # Initialize random one- and two-body integrals, and a constant
            one_body_integral = np.random.randn(2, 2)
            two_body_integral = np.random.randn(2, 2, 2, 2)
            constant = np.random.rand()

            # Define the MolecularHamiltonian
            mol_h = MolecularHamiltonian(one_body_integral, two_body_integral, constant)

            print(mol_h)

    """

    def __init__(
        self,
        one_body_integrals: np.ndarray,
        two_body_integrals: np.ndarray,
        constant_coeff: np.ndarray,
    ):

        self.one_body_integrals = one_body_integrals
        self.two_body_integrals = two_body_integrals
        self.constant_coeff = constant_coeff
        self.core_constant = 0

        self.active_indices = None
        self.occupied_indices = None

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def copy(self):
        """Copy class

        Returns:
            Self: Copy of class
        """

        return deepcopy(self)

    def __repr__(self):
        s = " MolecularHamiltonian(\n"
        s += f" - constant_coeff : {self.constant_coeff}\n"
        s += " - integrals shape\n"
        s += f"    * one_body_integrals : {self.one_body_integrals.shape}\n"
        s += f"    * two_body_integrals : {self.two_body_integrals.shape}\n)"

        return s

    @property
    def nqbits(self):
        "Compute number of qubits from the one body integral."
        return self.one_body_integrals.shape[0] * 2

    def transform_basis(self, transformation_matrix: np.ndarray) -> "MolecularHamiltonian":
        r"""
        Change one and two body integrals (indices p, q...) to new basis (indices i, j...)
        using transformation U such that

        .. math::

            \hat{c}_{i}=\sum_{q}U_{qi}c_{q}

        i.e

        .. math::

            \hat{I}_{ij} =\sum_{pq}U_{pi}I_{pq}U_{jq}^{\dagger}

            \hat{I}_{ijkl}=\sum_{pqrs}U_{pi}U_{qj}I_{pqrs}U_{kr}^{\dagger}U_{ls}^{\dagger}

        Args:
            transformation_matrix (np.array): transformation matrix :math:`U`

        Returns:
            molecular_hamiltonian (MolecularHamiltonian): MolecularHamiltonian updated to the new basis.

        """

        integrals = transform_integrals_to_new_basis(self.one_body_integrals, self.two_body_integrals, transformation_matrix)

        return MolecularHamiltonian(integrals[0], integrals[1], self.constant_coeff)

    def select_active_space(
        self,
        noons: List[float],
        n_electrons: int,
        threshold_1: Optional[float] = 2.0e-2,
        threshold_2: Optional[float] = 1.0e-3,
    ) -> Tuple["MolecularHamiltonian", List[int], List[int]]:
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
            noons (List[float]): the natural-orbital occupation numbers :math:`n_i`, sorted
                in descending order (from high occupations to low occupations)
            n_electrons (int): The number of electrons :math:`N_e`.
            threshold_1 (Optional[float]): The upper threshold :math:`\varepsilon_1` on
                the NOON of an active orbital.
            threshold_2 (Optional[float]): The lower threshold :math:`\varepsilon_2` on
                the NOON of an active orbital.

        Returns:
            Tuple[MolecularHamiltonian, List[int], List[int]]:
                - the molecular Hamiltonian in active space :math:`H^{(a)}`
                - the list of indices corresponding to the active orbitals, :math:`\mathcal{A}`
                - the list of indices corresponding to the occupied orbitals, :math:`\mathcal{O}`
        """

        active_indices, occupied_indices = select_active_orbitals(
            noons=noons,
            n_electrons=n_electrons,
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

    def get_electronic_hamiltonian(self) -> ElectronicStructureHamiltonian:
        r"""
        Converts the MolecularHamiltonian to an ElectronicStructureHamiltonian. To do so, it converts from :math:`I_{uv},I_{uvwx}`
        to :math:`h_{pq},h_{pqrs}`, with

        .. math::

            h_{u\sigma, v\sigma'} = I_{u, v} \delta_{\sigma, \sigma'}

            h_{u\sigma_1, v\sigma_2, w\sigma_2', x\sigma_1'} =  I_{uvwx} \left((1-\delta_{\sigma,\sigma'}) + \delta_{\sigma,\sigma'} (1-\delta_{u,v})(1-\delta_{w,x})   \right)

        and where the one- and two-body integrals are defined as:

        .. math::

            I_{uv}\equiv(u|h|v)=\int\mathrm{d}r\phi_{u}^{*}(r)T\phi_{v}(r)

        .. math::

            I_{uvwx}\equiv(ux|vw)=\iint\mathrm{d}r_{1}\mathrm{d}r_{2}\phi_{u}^{*}(r_{1})\phi_{x}(r_{1})v(r_{12})\phi_{v}^{*}(r_{2})\phi_{w}(r_{2})

        with :math:`T` (resp. :math:`v`) the one- (resp. two-) body potentials,
        and :math:`\phi_u(r)` is the molecular orbital wavefunction.

        The :math:`h` integrals are used to construct hamiltonians of the ElectronicStructureHamiltonian type.

        Returns:
            :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian` Electronic structure hamiltonian.

        """

        hpq, hpqrs = convert_to_h_integrals(self.one_body_integrals, self.two_body_integrals)

        h_electronic = ElectronicStructureHamiltonian(
            hpq,
            hpqrs,
            constant_coeff=self.constant_coeff + self.core_constant,
        )

        return h_electronic


class MoleculeInfo(object):
    r"""MoleculeInfo helper class. This class is a even higher level version of the
    :class:`~qat.fermion.chemistry.wrapper.MolecularHamiltonian`.

    Args:
        hamiltonian (MolecularHamiltonian): The MolecularHamiltonian of the studied molecule.
        n_electrons (int): Number of electrons.
        noons (Union[np.ndarray, List[float]]): Natural orbital occupation number.
        orbital_energies (np.ndarray): Orbital energies.

    Attributes:
        nqbits (int): The total number of qubits.
        one_body_integrals (np.ndarray): One-body integrals :math:`I_{uv}`.
        two_body_integrals (np.ndarray): Two-body integrals :math:`I_{uvwx}`.
        constant_coeff (np.ndarray): Constant coefficient :math:`r` (core repulsion).
        hamiltonian (MolecularHamiltonian): The :class:`~qat.fermion.chemistry.wrapper.MolecularHamiltonian` of the studied molecule.
        n_electrons (int): Number of electrons.
        noons (Union[np.ndarray, List[float]]): Natural orbital occupation number.
        orbital_energies (np.ndarray): Orbital energies.

    Example:

        .. run-block:: python

            import numpy as np
            from qat.fermion.chemistry import MolecularHamiltonian, MoleculeInfo

            # For illustration purpose, initialize random one- and two-body integrals, and a constant
            one_body_integral = np.random.randn(2, 2)
            two_body_integral = np.random.randn(2, 2, 2, 2)
            constant = np.random.rand()
            noons = list(np.random.randn(10))
            orbital_energies = list(np.random.randn(10))

            # Define the MolecularHamiltonian
            mol_h = MolecularHamiltonian(one_body_integral, two_body_integral, constant)

            # Define MoleculeInfo
            molecule = MoleculeInfo(
                mol_h,
                n_electrons=4,
                noons=noons,
                orbital_energies=orbital_energies
            )

            print(molecule)

    """

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

    def __repr__(self):
        """
        __repr__ method
        """

        h_str = self.hamiltonian.__repr__().replace("*", "**").replace("-", "*")

        s = "MoleculeInfo(\n"
        s += " - MolecularHamiltonian(\n"

        for st in h_str.splitlines()[1:]:
            s += f"   {st}\n"

        s += f" - n_electrons = {self.n_electrons}\n"
        s += f" - noons = {self.noons}\n"
        s += f" - orbital energies = {self.orbital_energies}\n"
        s += ")"
        return s

    @property
    def one_body_integrals(self) -> np.ndarray:
        """Getter for the one body integrals in the hamiltonian.

        Returns:
            np.ndarray: One body integrals :math:`I_{uv}`.
        """
        return self.hamiltonian.one_body_integrals

    @property
    def two_body_integrals(self) -> np.ndarray:
        """Getter for the two body integrals in the hamiltonian.

        Returns:
            np.ndarray: Two body integrals :math:`I_{uvwx}`.
        """
        return self.hamiltonian.two_body_integrals

    @property
    def constant_coeff(self) -> np.ndarray:
        """Getter for the constant coefficient in the hamiltonian.

        Returns:
            np.ndarray: Constant coefficient :math:`r` (core repulsion).
        """
        return self.hamiltonian.constant_coeff

    @property
    def nqbits(self):
        """
        Compute number of qubits from the one body integral.
        """
        return self.hamiltonian.nqbits

    def copy(self) -> "MolecularHamiltonian":
        """
        Copy the MoleculeInfo class.
        """
        return deepcopy(self)

    def restrict_active_space(
        self,
        threshold_1: Optional[float] = 2.0e-2,
        threshold_2: Optional[float] = 1.0e-3,
    ):
        r"""Same method as the :class:`~qat.fermion.chemistry.wrapper.MolecularHamiltonian` method
        :meth:`~qat.fermion.chemistry.wrapper.MolecularHamiltonian.select_active_space`, except it also modifies
        all the molecule parameters accordingly (NOONs, orbital energies, and number of electrons).

        For more information, see :meth:`~qat.fermion.chemistry.wrapper.MolecularHamiltonian.select_active_space`
        documentation.

        Args:
            threshold_1 (Optional[float]): The upper threshold :math:`\varepsilon_1` on
                the NOON of an active orbital.
            threshold_2 (Optional[float]): The lower threshold :math:`\varepsilon_2` on
                the NOON of an active orbital.

        """
        (
            self.hamiltonian,
            active_indices,
            occupied_indices,
        ) = self.hamiltonian.select_active_space(self.noons, self.n_electrons, threshold_1, threshold_2)

        self._update_molecule_active(active_indices, occupied_indices)

    def _update_molecule_active(self, active_indices: List[int], occupied_indices: List[int]):
        """
        Update MoleculeInfo attributes depending on the input active space indices and occupied indices.

        Args:
            active_indices (List[int]): List of active indices
            occupied_indices (List[int]): List of occupied indices

        """

        self.noons = [self.noons[idx] for idx in active_indices]
        self.orbital_energies = [self.orbital_energies[idx] for idx in active_indices]
        self.n_electrons = self.n_electrons - 2 * len(occupied_indices)

    def _get_attr_dict(self) -> Dict:
        """
        Generate attributes dictionary (needed for inclusion of MolecularHamiltonian attributes).

        Returns:
            Dict: Dictionary containing MoleculeInfo and MolecularHamiltonian attributes.

        """

        d = self.__dict__.copy()
        d.update(
            {
                "one_body_integrals": self.one_body_integrals,
                "two_body_integrals": self.two_body_integrals,
                "constant_coeff": self.constant_coeff,
            }
        )

        return d
