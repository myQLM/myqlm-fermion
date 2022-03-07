import numpy as np

from qat.plugins import ZeroNoiseExtrapolator
from qat.fermion.circuits import shallow
from qat.fermion.transforms import transform_to_jw_basis
from qat.fermion.hamiltonians import ElectronicStructureHamiltonian

from qat.hardware import make_depolarizing_hardware_model

from qat.qpus import NoisyQProc

np.random.seed(0)
eps1 = 0.0016
eps2 = 0.006
hw_model = make_depolarizing_hardware_model(
    eps1=eps1, eps2=eps2, depol_type="randomizing", correl_type="multi_qubit"
)
noisy_qpu = NoisyQProc(hardware_model=hw_model,
                       sim_method="deterministic-vectorized")

circ = shallow()
theta = np.random.random(8)
bd_circ = circ.bind_variables({r"\theta_{%i}" % i: theta[i] for i in range(8)})

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

job = bd_circ.to_job(observable=obs)


def test_linear_ZNE_emb_4qb():
    stack = ZeroNoiseExtrapolator() | noisy_qpu
    res = stack.submit(job)  # deterministic
    np.testing.assert_almost_equal(res.value, 1.4846479157570722, decimal=12)


def test_exponential_ZNE_emb_4qb():
    stack = ZeroNoiseExtrapolator(extrap_method="exponential") | noisy_qpu
    res = stack.submit(job)  # deterministic
    np.testing.assert_almost_equal(res.value, 1.48462310330087, decimal=12)
