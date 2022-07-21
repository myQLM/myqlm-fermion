from typing import List
import numpy as np
from tqdm.rich import tqdm
from qat.plugins import Junction
from qat.core import Result, Observable, Job
from qat.lang.AQASM import Program, RX, RY, RZ, H, CNOT


class AdaptVQEPlugin(Junction):

    r"""
    Adaptive ansatz plugin constructs an ansatz by selecting operators from a given pool of operators. The method is based
    on `Grimsley et al. article <https://www.nature.com/articles/s41467-019-10988-2.pdf>`_

        Args:
            operator_pool (List[Observable]): List of operators to choose from. The pool of commutators is internally constructed
            from this list.
            n_iterations (int, optional): Maximum number of iteration to perform. Defaults to 300.
            early_stopper (float, optional): Loss value for which the run is stopped. Defaults to 1e-6.
    """

    def __init__(
        self,
        operator_pool: List[Observable],
        n_iterations: int = 300,
        early_stopper: float = 1e-6,
    ):

        self.pool = operator_pool
        self.n_iterations = n_iterations
        self.early_stopper = early_stopper

        super().__init__(collective=False)

    def _compute_commutators(self, observable: Observable):
        """Compute commutators between pool operators and the observable.

        Args:
            job (Job): Job.

        Returns:
            list: List of Observable
        """
        return [observable | op for op in self.pool]

    @staticmethod
    def _grow_ansatz(operator: Observable, iter_num: int) -> Program:
        r"""
        Make a parameterized Trotter slice corresponding to \exp(-i \theta_i O)

        Args:
            op (Observable): Operator O (Hermitian).
            iter_num (int): Index i of parameter theta_i.

        Returns:
            Program: The corresponding program.

        """

        prog = Program()
        var = prog.new_var(float, "theta_" + str(iter_num))
        qbits = prog.qalloc(operator.nbqbits)

        if len(operator.terms) == 1 and len(operator.terms[0].qbits) == 1:

            term = operator.terms[0]
            pauli_string = term.op
            coeff = term.coeff.real
            pos = term.qbits[0]

            if pauli_string == "X":
                prog.apply(RX(2 * coeff * var), qbits[pos])

            elif pauli_string == "Y":
                prog.apply(RY(2 * coeff * var), qbits[pos])

            elif pauli_string == "Z":
                prog.apply(RZ(2 * coeff * var), qbits[pos])

            return prog

        for i, _ in enumerate(operator.terms):

            pauli_string = operator.terms[i].op
            list_qbits = operator.terms[i].qbits
            coeff = operator.terms[i].coeff.real

            # add RX(np.pi/2) for Y-gates and H for X-gates
            for current_pauli_op, current_qbit in zip(pauli_string, list_qbits):

                if current_pauli_op == "Y":
                    prog.apply(RX(np.pi / 2), qbits[current_qbit])

                elif current_pauli_op == "X":
                    prog.apply(H, qbits[current_qbit])

            # add CNOT gates
            for j in range(len(pauli_string) - 1):

                current_qbit = list_qbits[j]
                next_qbit = list_qbits[j + 1]
                prog.apply(CNOT, qbits[current_qbit], qbits[next_qbit])

            # add RZ-gate
            prog.apply(RZ(2 * coeff * var), qbits[next_qbit])

            # add CNOT gates back
            for j in range(len(pauli_string) - 1, 0, -1):

                current_qbit = list_qbits[j]
                previous_qbit = list_qbits[j - 1]
                prog.apply(CNOT, qbits[previous_qbit], qbits[current_qbit])

            # add RX(-np.pi/2) for Y-gates and H for X-gates back
            for current_pauli_op, current_qbit in zip(pauli_string, list_qbits):

                if current_pauli_op == "Y":
                    prog.apply(RX(-np.pi / 2), qbits[current_qbit])

                elif current_pauli_op == "X":
                    prog.apply(H, qbits[current_qbit])

        return prog

    def run(self, job: Job, _):

        # Get circuit
        circuit = job.circuit
        # Initialize Result container
        result = Result()

        # Initialize registers
        energy_gradients = []
        energy_trace = []
        operator_idx = []

        # Compute commutators
        commutators = self._compute_commutators(job.observable)

        # Iterate over number of input number of iterations
        for _ in range(self.n_iterations):

            # Loop over operator pool and find the one with biggest energy gradient
            for commutator in commutators:

                val = self.execute(circuit.to_job(observable=commutator)).value
                energy_gradients.append(val)

            # If all energy gradients are equal, pick randomly from the pool one of the operators
            energies_are_equal = all(item == energy_gradients[0] for item in energy_gradients)

            if energies_are_equal:
                op_ind = np.random.choice(np.arange(0, len(energy_gradients)))

            # Else pick the one which changes energy the most
            else:
                op_ind = np.argmax(np.abs(energy_gradients))

            operator_idx.append(op_ind)

            # Grow ansatz
            current_ansatz = self._grow_ansatz(self.pool[op_ind], 1)
            circuit += current_ansatz.to_circ()

            # Define the variational Job to optimize
            job = circuit.to_job(observable=job.observable)

            # Optimize the parameters
            result = self.execute(circuit.to_job(observable=job.observable))

            # Store current energy
            energy_trace += eval(result.meta_data["optimization_trace"])

            # Break ADAPT-VQE iterations if early stopping conditions are met
            if len(energy_trace) > 1:

                delta = energy_trace[-1] - energy_trace[-2]

                if delta < self.early_stopper:
                    break

        result.meta_data = dict()
        result.meta_data["operator_order"] = str(operator_idx)
        result.meta_data["optimization_trace"] = str(energy_trace)

        return result
