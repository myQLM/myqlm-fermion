# -*- coding: utf-8 -*-
"""
VQE function for backward compatibility
"""

from typing import Callable, List, Optional, Tuple
import numpy as np

from qat.lang.AQASM import Program

from .hamiltonians import SpinHamiltonian


def VQE(
    hamiltonian: SpinHamiltonian,
    optimizer,
    ansatz_routine: Callable,
    theta0: np.ndarray,
    qpu,
    n_shots: Optional[List[int]] = None,
) -> Tuple[float, list, int, List[float]]:
    r"""
    This function implements the Variational Quantum Eigen solver i.e., it first prepares the variational ansatz and measures the
    energy using a quantum processing unit (QPU), and then using a classical optimizer, finds the parameters of the ansatz that
    minimize the energy of the Hamiltonian.

    Args:
        hamiltonian (SpinHamiltonian): Hamiltonian for which the ground state is to be estimated
        optimizer (Optimizer): The optimization algorithm (a function) and its own parameters (either args or kwargs).
        ansatz_routine (Callable): Function of one list with all parameters to optimize, must return a QRoutine which
            corresponds to a ket.
        theta0 (np.ndarray): Initial list of parameters to optimize.
        qpu (QPU): Quantum process unit used. It can be get_qpu_server() (from qat.linalg import get_qpu_server) for ideal
            simulation or get_noisy_qpu_server(parameters) for noisy simulation for instance.
        n_shots (Optional[List[int]]): Two values which are either int or 0 (infinite number of shots). The first one determines the
        number of sample to measure one mean value for the optimisation. The second one is used to calculate the final energy. The
        bigger n_shots is, the more accurate the measurement of the mean value.

    Returns:
        float: Minimum energy.
        list: Optimized parameters.
        int: Number evaluation function.
        List[float]: Successive values of energy.

    Note:
        This high-level function is there just to maintain backward compatibility.

    """

    if n_shots is None:
        n_shots = [0, 0]

    def fun(theta, n_shots_internal):
        prog = Program()
        reg = prog.qalloc(hamiltonian.nbqbits)
        prog.apply(ansatz_routine(theta), reg)
        job = prog.to_circ().to_job(job_type="OBS", observable=hamiltonian, nbshots=n_shots_internal)
        res = qpu.submit(job)
        return res.value

    theta, energy, _, _ = optimizer(lambda theta: fun(theta, n_shots[0]), theta0)

    return energy, theta, None, None
