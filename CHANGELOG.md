Changelog
---------

Various improvements have been made in `MyQLM-fermion`. This includes new features, as well as many optimizations, bugfixes and
quality of life improvements.

### Main updates

*   The module is now called `qat.fermion` (replaces `qat.dqs`).
*   The `qchem` submodule has been renamed `chemistry` (`qat.fermion.chemistry`).
*   `qat.fermion` is fully compatible with PySCF 2.0 version.
*   The `SpinHamiltonian` and `FermionHamiltonian`classes have been rewritten:
    - `FermionHamiltonian` now has a `to_spin()` method.
    - Fermionic algebraic operations are now possible between `FermionHamiltonian` and/or `ElectronicStructureHamiltonian`.
    - Wick ordering and subsequent simplifications are now automatic for fermionic Hamiltonians.
    - It is now possible to cast `FermionHamiltonian` to an `ElectronicStructureHamiltonian`, using the `FermionHamiltonian.to_electronic()` method.

> Note : Since the `ElectronicStructureHamiltonian` inherits from `FermionHamiltonian`, any method available for `FermionHamiltonian` is available for `ElectronicStructureHamiltonian`.

*   Many improvements have been made to the chemistry module:
    *   the class `MolecularHamiltonian` has been added. It allows for easier basis changes and active space selection. It is also useful for easier transformation into `ElectronicStructureHamiltonian`.
    *   the class `MoleculeInfo` has been added to help with atomic and molecular computations.

A set of new plugins is now available:

* AdaptVQEPlugin: This plugin implements the ADAPT-VQE algorithm, to efficiently build an ansatz from a pool of operators; 
* GradientDescentOptimizer: Allows for natural gradient descent-based optimizations; 
* SequentialOptimizer: An implementation of the quantum-classical hybrid sequential minimal optimization method introduced by (Nakanashi et. al.)[https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043158] has been added; 
* ZeroNoiseExtrapolator: A plugin helping with multiqubit gate noise mitigation.

### Other improvements

*   `qat.dqs.fermionic_util` functions have now been included in `qat.fermion.util`.
    *   function `exact_eigen_energies` has been deleted.
    *   function `fermionic_hamiltonian_exponential` has been deleted.
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
