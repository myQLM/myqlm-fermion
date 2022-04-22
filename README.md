MyQLM fermion module
=============================

`MyQLM-fermion` is the open source version of the soon to be deprecated `DQS` module of the QLM, the digital quantum simulations 
module.

This module contains tools specific to spin and fermionic systems. It includes, among other things:
- various objects to describe fermionic and spin Hamiltonians,
- various transformations between fermionic and spin representations,
- a variational quantum eigensolver (VQE), as well as the fermionic and qubit version of ADAPT-VQE,
- a trotterization module,
- a quantum phase estimation module,
- various tools for electronic structure study,
- ...

Installation
----------------

You can install the fermion module with pip directly from the internet (not supported for now):
```shell
pip install myqlm-fermion
```
or install from the sources directly :
```python
git clone https://github.com/myQLM/myqlm-fermion.git
cd myqlm-fermion
pip install .
```

Documentation
-------------------

This module is part of the MyQLM package, whose documentation can be found [here](https://myqlm.github.io/).
To access directly MyQLM-fermion documentation, [click here](https://myqlm.github.io/myqlm_specific/fermion.html).

Note: The documentation website has not been updated yet.

Changelog
---------

Various improvements have been made in `MyQLM-fermion`.

### Main update

*   The module is now called `qat.fermion` (replaces `qat.dqs`).
*   The `qchem` submodule has been renamed `chemistry` (`qat.fermion.chemistry`).
*   `qat.fermion` is fully compatible with PySCF 2.0 version.
*   The `SpinHamiltonian` and `FermionHamiltonian`classes are now deprecated. They have been replaced by the flexible class `Hamiltonian` which can be in both spin or fermionic representation. The `Hamiltonian` class contains new methods:
    *   The `.htype` method allows to check the current representation of the `Hamiltonian`,
    *   The `.to_spin()` method allows for the direct conversion of a fermionic type `Hamiltonian` to a `Hamiltonian` in spin representation.

> Note : See `spin_fermion_transforms.ipynb` in `myqlm-fermion/doc/notebooks/` for more informations on this new `Hamiltonian` class.

*   The `ElectronicStructureHamiltonian` inherits from the `Hamiltonian` class. The above methods are also valid for `ElectronicStructureHamiltonian`. Additionally, `ElectronicStructureHamiltonian` now contains an `.eigenenergies()` method and an `.exponential()` method, see `qat.fermion.hamiltonians.ElectronicStructureHamiltonian` docstring.
*   Many improvements have been made to the UCC submodule.
*   New classes specific to atomic and molecular studies have been added:
    *   the class `MolecularHamiltonian` has been added. It allows for easier basis changes and active space selection.
    *   the class `MoleculeInfo` has been added to help with atomic and molecular computations.

> For more information on the UCC changes and the new helper classes, see Jupyter notebook `vqe_ucc_example_1_h2.ipynb` and `vqe_ucc_example_2_lih.ipynb` in `myqlm-fermion/doc/notebooks/`.

### Other improvements

*   `qat.dqs.fermionic_util` functions have now been included in `qat.fermion.util`.
    *   function `exact_eigen_energies` has been deleted (replaced by the `eigenenergies()` method in `ElectronicStructureHamiltonian`).
    *   function `fermionic_hamiltonian_exponential` has been deleted  (replaced by the `exponential()` method in `ElectronicStructureHamiltonian`).
*   `qat.dqs.impurity` models have been relocated in `qat.fermion.hamiltonians`.
*   `qat.dqs.ansatz_circuits` has been renamed `qat.fermion.circuits`.
*   In `qat.fermion.chemistry.ucc` (previously `qat.dqs.qchem.ucc`):
    *   `build_ucc_ansatz` has been deprecated. It has been relocated to `qat.fermion.chemistry.ucc_deprecated`, but the method `construct_ucc_ansatz` should be used instead. The underlying lower level methods have been clarified and can be used as well.
    *   `compute_active_space_integrals` has been split into `_compute_active_space_constant` and `compute_active_space_integrals`.
    *   `build_cluster_ops` is now a private method.
    *   `select_active_orbitals` default threshold values have been updated.
    *   `init_uccsd` is now a private method, and returns only the trial Mollet-Plesset 2nd order guess initial parameter.
    *   The `get_initial_params_and_cluster_ops` function has been split into 3 separate functions for better clarity:
        *   `guess_init_params` to get the initial parameter guess computed via 2nd order Mollet-Plesset perturbation theory,
        *   `get_hf_ket` to get the Hartree-Fock state,
        *   `get_cluster_ops` to get the cluster operator list.
*   `select_excitation_operators` and `get_cluster_ops` have been updated. The deprecated versions are located in `qat.fermion.chemistry.ucc_deprecated`.
*   `get_active_space_hamiltonian` has been deprecated. It can still be found in `qat.fermion.chemistry.ucc_deprecated`.
