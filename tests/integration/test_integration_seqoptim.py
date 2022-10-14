# -*- coding: utf-8 -*-
"""
Integration test for sequential optimization plugin
"""

import numpy as np

from qat.plugins import SeqOptim
from qat.qpus import get_default_qpu

from qat.fermion.circuits import make_shallow_circ
from qat.fermion.hamiltonians import ElectronicStructureHamiltonian
from qat.fermion.transforms import transform_to_jw_basis


def test_VQE_SeqOptim_emb_4qb():
    np.random.seed(1)
    x0 = np.random.random(8)
    stack = SeqOptim(x0=x0) | get_default_qpu()
    U = 1
    mu = U / 2
    V = 1
    eps = 1
    hpq = np.array([[mu, V, 0, 0], [V, eps, 0, 0], [0, 0, mu, V], [0, 0, V, eps]])
    hpqrs = np.zeros((4, 4, 4, 4))
    hpqrs[0, 2, 0, 2] = -1
    hpqrs[2, 0, 2, 0] = -1
    hamilt = ElectronicStructureHamiltonian(hpq=hpq, hpqrs=hpqrs)
    obs = transform_to_jw_basis(hamilt)  # spin ordering
    circ = make_shallow_circ()
    job = circ.to_job(observable=obs)
    res = stack.submit(job)  # deterministic
    np.testing.assert_almost_equal(res.value, -0.28053014718543934, decimal=12)
