# -*- coding: utf-8 -*-
"""
ADAPT-VQE Plugin
"""

import warnings
from typing import List
from tqdm.auto import tqdm
import numpy as np

from qat.core.junction import Junction
from qat.core import Result, Observable, Job
from qat.lang.AQASM import Program, RX, RY, RZ, H, CNOT


class AdaptVQEPlugin(Junction):
    r"""
    Plugin implementation of the ADAPT-VQE algorithm, which builds ansatze by selecting operators from a given pool of operators.
    The method is based on `Grimsley et al. article <https://www.nature.com/articles/s41467-019-10988-2.pdf>`_.

    Args:
        operator_pool (List[Observable]): List of operators to choose from.
            The pool of commutators is internally constructed from this list.
        n_iterations (int, optional): Maximum number of iteration to perform. Defaults to 300.
        commutators (List[Union[Observable, SpinHamiltonian]): List of commutators to use when computing the energy gradients.
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
        self.commutators = None

        super().__init__(collective=False)

    def _compute_commutators(self, observable: Observable):
        """Compute commutators between pool operators and the observable.

        Args:
            job (Job): Job.

        Returns:
            list: List of Observable
        """

        return [observable | op for op in tqdm(self.pool, desc="Computing commutators...")]

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
        var = prog.new_var(float, f"theta_{iter_num}")
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

            # Add RX(np.pi/2) for Y-gates and H for X-gates
            for current_pauli_op, current_qbit in zip(pauli_string, list_qbits):

                if current_pauli_op == "X":
                    prog.apply(H, qbits[current_qbit])

                elif current_pauli_op == "Y":
                    prog.apply(RX(np.pi / 2), qbits[current_qbit])

            # Add CNOT gates
            for j in range(len(pauli_string) - 1):

                current_qbit = list_qbits[j]
                next_qbit = list_qbits[j + 1]
                prog.apply(CNOT, qbits[current_qbit], qbits[next_qbit])

            # Add RZ-gate
            prog.apply(RZ(2 * coeff * var), qbits[next_qbit])

            # Add CNOT gates back
            for j in range(len(pauli_string) - 1, 0, -1):

                current_qbit = list_qbits[j]
                previous_qbit = list_qbits[j - 1]
                prog.apply(CNOT, qbits[previous_qbit], qbits[current_qbit])

            # Add RX(-np.pi/2) for Y-gates and H for X-gates back
            for current_pauli_op, current_qbit in zip(pauli_string, list_qbits):

                if current_pauli_op == "X":
                    prog.apply(H, qbits[current_qbit])

                elif current_pauli_op == "Y":
                    prog.apply(RX(-np.pi / 2), qbits[current_qbit])

        return prog

    def run(self, job: Job, _) -> Result:
        """Execute the job

        Args:
            job (Job): Job.

        Returns:
            Result: Result
        """

        # Get circuit
        circuit = job.circuit

        # Initialize Result container
        result = Result()

        # Initialize registers
        energy_trace = []
        operator_idx = []

        # Compute commutators
        commutators = self._compute_commutators(job.observable)

        pbar = tqdm(range(self.n_iterations))
        # Iterate over number of input number of iterations
        for _ in pbar:

            # Step energy register
            energy_gradients = []

            # Loop over operator pool and find the one with biggest energy gradient
            pbar.set_description("Computing energy gradients...")
            for commutator in commutators:

                val = self.execute(circuit.to_job(observable=commutator)).value
                energy_gradients.append(val)

            # If all energy gradients are equal, pick randomly from the pool one of the operators
            energies_are_equal = all(item == energy_gradients[0] for item in energy_gradients)

            if energies_are_equal:

                # # If we have converged, energy gradient will remain at zero for all values
                if energy_gradients[0] == 0:
                    warnings.warn(
                        "All energy gradients are equal to zero for given operator pool. Ending calculation.", stacklevel=2
                    )
                    break

                # If energy gradients are equal, pick randomly one of them
                op_ind = np.random.choice(np.arange(0, len(commutators) - 1))

            # Else pick the one which changes energy the most (if 2 or more are equal, pick the first one)
            else:
                op_ind = np.argmax(np.abs(energy_gradients))

            operator_idx.append(op_ind)

            # Grow ansatz
            pbar.set_description("Growing ansatz...")

            current_ansatz = self._grow_ansatz(self.pool[op_ind], 1)
            circuit += current_ansatz.to_circ()

            # Define the variational Job to optimize
            job = circuit.to_job(observable=job.observable)

            # Optimize the parameters
            result = self.execute(circuit.to_job(observable=job.observable))

            # Store current energy
            # pylint: disable=eval-used
            energy_trace += eval(result.meta_data["optimization_trace"])

        result.meta_data = {}
        result.meta_data["operator_order"] = str(operator_idx)
        result.meta_data["optimization_trace"] = str(energy_trace)

        return result
