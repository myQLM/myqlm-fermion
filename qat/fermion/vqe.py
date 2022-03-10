from qat.lang.AQASM import Program


def VQE(
    hamiltonian, optimizer, ansatz_routine, theta0, qpu, n_shots=[0, 0], display=False
):
    r"""
    This function implements the Variational Quantum Eigen solver i.e.,
    it first prepares the variational ansatz and measures the energy using
    a quantum processing unit (QPU), and then using a classical optimizer,
    finds the parameters of the ansatz that minimize the energy of the Hamiltonian.

    Args:
        hamiltonian (Hamiltonian): hamiltonian for which
            the ground state is to be estimated
        optimizer (Optimizer): with 2 attributes : the optimization algorithm
            (a function) and its own parameters (either args or kwargs)
        ansatz_routine (function): function of one list with all parameters to
            optimize, must return a QRoutine which corresponds to a ket
        theta0 (numpy.array): initial list of parameters to optimize
        qpu (QPU): quantum process unit used. It can be get_qpu_server()
            (from qat.linalg import get_qpu_server) for ideal simulation or
            get_noisy_qpu_server(parameters) for noisy simulation for instance
        n_shots (list): two values which are either int or 0 (infinite number of shots).
            The first one determines the number of sample to measure one mean value
            for the optimisation. The second one is used to calculate the final energy.
            The bigger n_shots is, the more accurate the measurement of the mean value.
        ancilla_qubit(int): number of ancilla qubits used in the preparation state (if needed)

    Returns:
        float: minimum energy
        list: optimized parameters
        int: number evaluation function
        list(float): successive values of energy

    Note:
        This high-level function is there just to maintain backward compatibility.
    """

    def fun(theta, n_shots_internal):
        prog = Program()
        reg = prog.qalloc(hamiltonian.nbqbits)
        prog.apply(ansatz_routine(theta), reg)
        job = prog.to_circ().to_job(
            job_type="OBS", observable=hamiltonian, nbshots=n_shots_internal
        )
        res = qpu.submit(job)
        return res.value

    # optimized_params, nb_eval, energies = optimizer.make_calculation(lambda theta: fun(theta, n_shots[0]), theta0)
    # minimum_energy = fun(optimized_params, n_shots[1])

    theta, energy, k, theta_energy_list = optimizer(
        lambda theta: fun(theta, n_shots[0]), theta0
    )

    # if display:
    # print("Energy =", minimum_energy, "Optimized parameters =", optimized_params,
    #       "\n Number of function evaluations =", nb_eval)
    # print("Res = ", res)
    # print("Res = ", res)

    # return minimum_energy, optimized_params, nb_eval, energies
    return energy, theta, None, None
