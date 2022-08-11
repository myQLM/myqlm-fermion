import numpy as np

from qat.lang.AQASM import Program, X
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin, AdaptVQEPlugin

from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation
from qat.fermion.chemistry import MolecularHamiltonian, MoleculeInfo
from qat.fermion.chemistry.ucc import get_hf_ket, get_cluster_ops
from qat.fermion.transforms import transform_to_bk_basis, recode_integer, get_bk_code


def compute_cluster_ops(geometry: str, basis: str, charge: int, spin: int, thresholds=[0.02, 0.002]):

    threshold_1, threshold_2 = thresholds

    (
        rdm1,
        orbital_energies,
        nuclear_repulsion,
        n_electrons,
        one_body_integrals,
        two_body_integrals,
        _,
    ) = perform_pyscf_computation(geometry=geometry, basis=basis, spin=spin, charge=charge)

    # Define the molecular hamiltonian
    mol_h = MolecularHamiltonian(one_body_integrals, two_body_integrals, nuclear_repulsion)

    ## Active space selection
    # Compute the natural orbitals occupation numbers and the basis transformation matrix
    noons, basis_change = np.linalg.eigh(rdm1)

    # The noons should be in decreasing order
    noons = list(reversed(noons))

    # Since we reversed the noons, we have to flip the basis as well
    basis_change = np.flip(basis_change, axis=1)

    # Change the hamiltonian basis
    mol_h_new_basis = mol_h.transform_basis(basis_change)

    molecule = MoleculeInfo(
        mol_h_new_basis,
        n_electrons=n_electrons,
        noons=noons,
        orbital_energies=orbital_energies,
    )

    # Selection of the active space
    molecule.restrict_active_space(threshold_1=threshold_1, threshold_2=threshold_2)

    ## Computation of cluster operators $T$ and good guess $\vec{\theta}_0$

    # Compute the cluster operators
    cluster_ops = get_cluster_ops(molecule.n_electrons, noons=molecule.noons)

    # Define the initial Hartree-Fock state
    ket_hf_init = get_hf_ket(molecule.n_electrons, nqbits=molecule.nqbits)

    ## Encode to qubits: Fermion-spin transformation

    # Compute the ElectronicStructureHamiltonian
    H_active = molecule.hamiltonian.get_electronic_hamiltonian()

    # Transform the ElectronicStructureHamiltonian into a spin Hamiltonian
    H_active_sp = transform_to_bk_basis(H_active)

    # Express the cluster operator in spin terms
    cluster_ops_sp = [transform_to_bk_basis(t_o) for t_o in cluster_ops]

    # Encoding the initial state to new encoding
    hf_init_sp = recode_integer(ket_hf_init, get_bk_code(H_active_sp.nbqbits))

    return H_active_sp, cluster_ops_sp, hf_init_sp


def test_H2_molecule_adapt():

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    charge = 0
    spin = 0
    basis = "sto-3g"

    H_sp, cluster_ops_sp, hf_init_sp = compute_cluster_ops(geometry, basis, charge, spin)

    # Compute exact energy via diagonalization
    exact_energy = np.min(np.linalg.eigvalsh(H_sp.get_matrix()))

    # Hartree-Fock state preparation circuit
    prog = Program()
    reg = prog.qalloc(H_sp.nbqbits)

    # Initialize the Hartree-Fock state into the Program
    for j, char in enumerate(format(hf_init_sp, "0" + str(H_sp.nbqbits) + "b")):
        if char == "1":
            prog.apply(X, reg[j])

    circuit = prog.to_circ()

    job = circuit.to_job(observable=H_sp)

    adaptvqe_plugin = AdaptVQEPlugin(cluster_ops_sp, n_iterations=30)
    optimizer = ScipyMinimizePlugin(method="COBYLA", tol=1e-6, options={"maxiter": 500})
    qpu = get_default_qpu()

    stack = adaptvqe_plugin | optimizer | qpu

    result = stack.submit(job)

    np.testing.assert_almost_equal(exact_energy, result.value, decimal=2)


def test_lih_molecule_adapt():

    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.75))]
    basis = "6-31g"
    spin = 0
    charge = 0

    H_sp, cluster_ops_sp, hf_init_sp = compute_cluster_ops(geometry, basis, charge, spin)

    # Compute exact energy via diagonalization
    exact_energy = np.min(np.linalg.eigvalsh(H_sp.get_matrix()))

    # Hartree-Fock state preparation circuit
    prog = Program()
    reg = prog.qalloc(H_sp.nbqbits)

    # Initialize the Hartree-Fock state into the Program
    for j, char in enumerate(format(hf_init_sp, "0" + str(H_sp.nbqbits) + "b")):
        if char == "1":
            prog.apply(X, reg[j])

    circuit = prog.to_circ()

    job = circuit.to_job(observable=H_sp)

    adaptvqe_plugin = AdaptVQEPlugin(cluster_ops_sp, n_iterations=15)
    optimizer = ScipyMinimizePlugin(method="COBYLA", tol=1e-10, options={"maxiter": 200})
    qpu = get_default_qpu()

    stack = adaptvqe_plugin | optimizer | qpu

    result = stack.submit(job)

    np.testing.assert_almost_equal(exact_energy, result.value, decimal=2)
