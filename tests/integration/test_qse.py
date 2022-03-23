import numpy as np
from qat.core import Term
from qat.fermion import Hamiltonian
from qat.lang.AQASM import Program, RY, CNOT, RZ
from qat.qpus import NoisyQProc
from qat.plugins import SeqOptim
from qat.fermion.chemistry.qse import (
    apply_quantum_subspace_expansion,
    build_linear_pauli_expansion,
)

# we instantiate the Hamiltonian we want to approximate the ground state energy of
hamiltonian = Hamiltonian(2, [Term(1, op, [0, 1]) for op in ["XX", "YY", "ZZ"]])


# we construct the variational circuit (ansatz)
prog = Program()
reg = prog.qalloc(2)
theta = [prog.new_var(float, "\\theta_%s" % i) for i in range(3)]
RY(theta[0])(reg[0])
RY(theta[1])(reg[1])
RZ(theta[2])(reg[1])
CNOT(reg[0], reg[1])
circ = prog.to_circ()

# construct a (variational) job with the variational circuit and the observable
job = circ.to_job(observable=hamiltonian, nbshots=0)

# we now build a stack that can handle variational jobs
from qat.hardware import make_depolarizing_hardware_model

qpu = NoisyQProc(
    hardware_model=make_depolarizing_hardware_model(eps1=0.001, eps2=0.01),
    sim_method="deterministic-vectorized",
)
optimizer = SeqOptim(ncycles=10, x0=[0, 0.5, 0])
stack = optimizer | qpu

# we submit the job and print the optimized variational energy (the exact GS energy is -3)
result = stack.submit(job)
E_min = -3
print(
    "E(VQE) = %s (err = %s %%)"
    % (result.value, 100 * abs((result.value - E_min) / E_min))
)
e_vqe = result.value

# we use the optimal parameters found by VQE
opt_circ = circ.bind_variables(eval(result.meta_data["parameter_map"]))

expansion_operators = [
    Hamiltonian(2, [], 1.0),
    Hamiltonian(2, [Term(1.0, "ZZ", [0, 1])]),
]


def test_apply_quantum_subspace_expansion():

    e_qse = apply_quantum_subspace_expansion(
        hamiltonian, opt_circ, expansion_operators, qpu, return_matrices=False
    )

    np.testing.assert_equal(e_qse, -2.979211358273586)


def test_build_linear_pauli_expansion():

    expectation = [
        np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]]),
        np.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]),
        np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]),
        np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]),
        None,
    ]

    test = build_linear_pauli_expansion(["X", "Y", "Z", "Z", "I"], 1)
    test = [test[i].get_matrix() for i in range(len(test))]

    print(test)

    np.testing.assert_equal(test, expectation)
