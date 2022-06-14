#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy
import numpy as np

from qat.qpus import get_default_qpu
from qat.lang.AQASM import Program

from qat.fermion.hamiltonians import ElectronicStructureHamiltonian
from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation
from qat.fermion.chemistry.ucc import (
    convert_to_h_integrals,
)
from qat.fermion.chemistry.ucc_deprecated import (
    get_cluster_ops_and_init_guess,
    build_ucc_ansatz,
    get_active_space_hamiltonian,
)
from qat.fermion.transforms import recode_integer, transform_to_jw_basis, get_jw_code


def prepare_h2(use_pyscf=False, verbose=False):

    if use_pyscf:
        # geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.75))]
        # basis = "6-31g"
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        basis = "sto-3g"

        (
            rdm1,
            orbital_energies,
            nuclear_repulsion,
            nels,
            one_body_integrals,
            two_body_integrals,
        ) = perform_pyscf_computation(geometry=geometry, basis=basis, spin=0, charge=0, verbose=True)

        # get NOONs from 1-RDM (computed in CISD)
        noons = list(reversed(sorted(np.linalg.eigvalsh(rdm1))))
        if verbose:
            print("nels=", nels)
            print("I1=", one_body_integrals)
            print("I2=", two_body_integrals)
            print("orb energies", orbital_energies)
            print("nuc repulsion=", nuclear_repulsion)
            print("noons=", noons)
    else:
        nels = 2
        one_body_integrals = np.array([[-1.25246357, 0], [0, -0.475948715]])
        two_body_integrals = np.array(
            [
                [
                    [[0.674488766, 0], [0, 0.181288808]],
                    [[0, 0.181288808], [0.663468096, 0]],
                ],
                [
                    [[0, 0.663468096], [0.181288808, 0]],
                    [[0.181288808, 0], [0, 0.697393767]],
                ],
            ]
        )
        orbital_energies = np.array([-0.57797481, 0.66969867])
        nuclear_repulsion = 0.7137539936876182

        noons = np.array([1.9745399697399246, 0.025460030260075376])
    return (
        nels,
        one_body_integrals,
        two_body_integrals,
        orbital_energies,
        nuclear_repulsion,
        noons,
    )


def test_basic(use_pyscf=False, verbose=False):
    (
        nels,
        one_body_integrals,
        two_body_integrals,
        orbital_energies,
        nuclear_repulsion,
        noons,
    ) = prepare_h2()

    H_active, active_inds, occ_inds = get_active_space_hamiltonian(
        one_body_integrals,
        two_body_integrals,
        noons,
        nels,
        nuclear_repulsion,
        threshold_2=1e-3,
    )
    active_noons, active_orb_energies = [], []
    for ind in active_inds:
        active_noons.extend([noons[ind], noons[ind]])
        active_orb_energies.extend([orbital_energies[ind], orbital_energies[ind]])
    nb_active_els = nels - 2 * len(occ_inds)

    cluster_ops, theta_0, hf_init = get_cluster_ops_and_init_guess(nb_active_els, active_noons, active_orb_energies, H_active.hpqrs)

    # transform, code = transform_to_parity_basis, get_parity_code
    # transform, code = transform_to_bk_basis, get_bk_code
    transform, code = transform_to_jw_basis, get_jw_code

    H_active_sp = transform(H_active)
    nqbits = H_active_sp.nbqbits

    # expressing the cluster operator in spin terms
    cluster_ops_sp = [transform(t_o) for t_o in cluster_ops]

    # encoding the initial state to new encoding
    hf_init_sp = recode_integer(hf_init, code(nqbits))

    # Finally: construct_ucc_ansatz
    qrout = build_ucc_ansatz(cluster_ops_sp, hf_init_sp)

    # we define the cost function to be minimized (the energy)
    def fun(theta):
        qpu = get_default_qpu()
        prog = Program()
        reg = prog.qalloc(nqbits)
        prog.apply(qrout(theta), reg)
        circ = prog.to_circ()
        res = qpu.submit(circ.to_job(job_type="OBS", observable=H_active_sp))
        return res.value

    res = scipy.optimize.minimize(
        lambda theta: fun(theta),
        x0=theta_0,
        method="COBYLA",
        tol=1e-15,
        options={"maxiter": 1000},
    )
    np.testing.assert_almost_equal(res.fun, -1.1372701746609022, decimal=8)
    # FCI energy is -1.1372701746609022


def test_more_basic(use_pyscf=False, verbose=False):
    (
        nels,
        one_body_integrals,
        two_body_integrals,
        orbital_energies,
        nuclear_repulsion,
        noons,
    ) = prepare_h2(use_pyscf)
    hpq, hpqrs = convert_to_h_integrals(one_body_integrals, two_body_integrals)
    H = ElectronicStructureHamiltonian(hpq, hpqrs, constant_coeff=nuclear_repulsion)
    noons_full, orb_energies_full = [], []
    for ind in range(len(noons)):
        noons_full.extend([noons[ind], noons[ind]])
        orb_energies_full.extend([orbital_energies[ind], orbital_energies[ind]])

    cluster_ops, theta_0, hf_init = get_cluster_ops_and_init_guess(nels, noons_full, orb_energies_full, H.hpqrs)

    transform, code = transform_to_jw_basis, get_jw_code

    # expressing the Hamiltonian and cluster operator in spin terms
    # and encoding the initial state to new encoding
    H_sp = transform(H)
    cluster_ops_sp = [transform(t_o) for t_o in cluster_ops]
    hf_init_sp = recode_integer(hf_init, code(H_sp.nbqbits))

    # Finally: build_uccsd
    qrout = build_ucc_ansatz(cluster_ops_sp, hf_init_sp)

    # we define the cost function to be minimized (the energy)
    def fun(theta):
        qpu = get_default_qpu()
        prog = Program()
        reg = prog.qalloc(H_sp.nbqbits)
        prog.apply(qrout(theta), reg)
        circ = prog.to_circ()
        res = qpu.submit(circ.to_job(job_type="OBS", observable=H_sp))
        return res.value

    res = scipy.optimize.minimize(
        lambda theta: fun(theta),
        x0=theta_0,
        tol=1e-15,
        method="COBYLA",
        options={"maxiter": 1000},
    )
    np.testing.assert_almost_equal(res.fun, -1.1372701746609022, decimal=8)
    # FCI energy is -1.1372701746609022
