import numpy as np

from qat.lang.AQASM import H, RX, RY, CNOT, QRoutine, Program
from qat.qpus import get_default_qpu

from qat.fermion.hamiltonians import make_hubbard_model
from qat.fermion.naturalgradient.qfim_plugin import GradientMinimizePlugin
from qat.fermion.matchgates import RYX, RXY


def ansatz_circuit(theta):
    """Take a parameter theta and return the corresponding circuit"""
    Qrout = QRoutine()
    Qrout.apply(H, 0)
    Qrout.apply(RY(theta[0]), 0)
    Qrout.apply(CNOT, 0, 1)
    Qrout.apply(RX(theta[1]), 1)
    return Qrout


def ansatz_with_custom_RXX_gates(theta):
    """Take a parameter theta and return the corresponding circuit"""
    Qrout = QRoutine()
    Qrout.apply(H, 0)
    Qrout.apply(RXY(theta[0]), 0, 1)
    Qrout.apply(RY(theta[0]), 0)
    Qrout.apply(CNOT, 0, 1)
    Qrout.apply(RX(theta[1]), 1)
    Qrout.apply(RYX(theta[1]), 0, 1)

    return Qrout


# Define a Hubbard model to solve
U = 2.0
nqbit = 2
t_mat = np.zeros((1, 1))
hamiltonian = make_hubbard_model(t_mat, U, mu=U / 2).to_spin()

# Compute exact energy
eigvals, eigvecs = np.linalg.eigh(hamiltonian.get_matrix())
exact_energy = min(eigvals)

prog = Program()
reg = prog.qalloc(hamiltonian.nbqbits)
prog.apply(
    ansatz_circuit([prog.new_var(float, "\\theta_%s" % i) for i in range(hamiltonian.nbqbits)]),
    reg,
)
circ = prog.to_circ()

job = circ.to_job(job_type="OBS", observable=hamiltonian, nbshots=0)

linalg_qpu = get_default_qpu()

np.random.seed(1234)


def test_gradient_descent():
    """Test gradient descent plugin without using natural gradients"""

    variables = circ.get_variables()
    x0 = {variable: value for variable, value in zip(variables, np.random.uniform(0, 2 * np.pi, len(variables)))}

    gradient_descent = GradientMinimizePlugin(maxiter=100, lambda_step=0.2, natural_gradient=False, tol=1e-7, x0=x0)
    qpu = gradient_descent | linalg_qpu
    result = qpu.submit(job)

    np.testing.assert_almost_equal(result.value, exact_energy, decimal=4)


def test_natural_gradient_descent():
    """Test gradient descent plugin using natural gradients"""

    variables = circ.get_variables()
    x0 = {variable: value for variable, value in zip(variables, np.random.uniform(0, 2 * np.pi, len(variables)))}

    gradient_descent = GradientMinimizePlugin(maxiter=100, lambda_step=0.2, natural_gradient=True, tol=1e-7, x0=x0)
    qpu = gradient_descent | linalg_qpu
    result = qpu.submit(job)

    np.testing.assert_almost_equal(result.value, exact_energy, decimal=4)


def test_natural_gradient_descent_no_initialization():
    """Test gradient descent plugin with no user initialization of the parameters (random initialization)"""

    gradient_descent = GradientMinimizePlugin(maxiter=100, lambda_step=0.2, natural_gradient=True, tol=1e-7)
    qpu = gradient_descent | linalg_qpu
    result = qpu.submit(job)

    np.testing.assert_almost_equal(result.value, exact_energy, decimal=4)


def test_natural_gradient_with_custom_RXX_gates():
    """Test gradient descent plugin when circuit contains RYY, RXX,...etc custom gates."""

    prog = Program()
    reg = prog.qalloc(hamiltonian.nbqbits)
    prog.apply(
        ansatz_with_custom_RXX_gates([prog.new_var(float, "\\theta_%s" % i) for i in range(hamiltonian.nbqbits)]),
        reg,
    )
    circ = prog.to_circ()

    job = circ.to_job(job_type="OBS", observable=hamiltonian, nbshots=0)

    linalg_qpu = get_default_qpu()

    gradient_descent = GradientMinimizePlugin(maxiter=100, lambda_step=0.2, natural_gradient=False, tol=1e-7)
    qpu = gradient_descent | linalg_qpu
    result = qpu.submit(job)

    np.testing.assert_almost_equal(result.value, exact_energy, decimal=4)


def test_natural_gradient_with_custom_RXX_gates_energy_earlystop():
    """Test gradient descent plugin when circuit contains RYY, RXX,...etc custom gates."""

    prog = Program()
    reg = prog.qalloc(hamiltonian.nbqbits)
    prog.apply(
        ansatz_with_custom_RXX_gates([prog.new_var(float, "\\theta_%s" % i) for i in range(hamiltonian.nbqbits)]),
        reg,
    )
    circ = prog.to_circ()

    job = circ.to_job(job_type="OBS", observable=hamiltonian, nbshots=0)

    linalg_qpu = get_default_qpu()

    gradient_descent = GradientMinimizePlugin(
        maxiter=100, lambda_step=0.2, natural_gradient=False, tol=1e-7, stop_crit="energy_dist"
    )
    qpu = gradient_descent | linalg_qpu
    result = qpu.submit(job)

    np.testing.assert_almost_equal(result.value, exact_energy, decimal=4)
