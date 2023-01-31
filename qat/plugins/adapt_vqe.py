# -*- coding: utf-8 -*-
"""
ADAPT-VQE Plugin
"""

import warnings
from typing import List
from tqdm.auto import tqdm
import copy
import numpy as np

from qat.core.junction import Junction
from qat.core import Result, Observable, Job
from qat.lang.AQASM import Program, RX, RY, RZ, H, CNOT


class AdaptVQEPlugin(Junction):
    r"""
    Plugin implementation of the ADAPT-VQE algorithm, which builds ansatze by selecting operators :math:`\tau_k` from a user-defined pool of operators.
    Once an operator is chosen, a parameterized gate :math:`\exp(\theta_k \tau_k)` is added to the circuit.
    The method is based on the `Grimsley et al. <https://www.nature.com/articles/s41467-019-10988-2.pdf>`_ publication.


    Args:
        operator_pool (List[Observable]): List of operators :math:`\tau_k` to choose from (they should be antihermitian).
            The pool of commutators is either given by the user or internally constructed from this list.
        n_iterations (int, optional): Maximum number of iterations to perform. Defaults to 300.
        tol_vanishing_grad (float, optional): threshold value of the norm-2 of the gradient vector under which 
            to stop the computation. Defaults to 1e-3.
        commutators (List[Observable], optional): List of commutators between the observable and an operator from 
            the pool, whose expectation values yield the gradient. Defaults to None, in which case it is constructed
            when the plugin is run.
    """

    def __init__(
        self,
        operator_pool: List[Observable],
        n_iterations: int = 300,
        tol_vanishing_grad : float = 1e-3,
        commutators = None
    ):

        self.pool = operator_pool
        self.n_iterations = n_iterations
        self.tol_vanishing_grad = tol_vanishing_grad
        self.commutators = commutators

        super().__init__(collective=False)

    @staticmethod
    def _compute_commutators(observable: Observable, pool:List[Observable]):
        """Compute commutators between pool operators and the observable.

        Args:
            observable (Observable): The Observable whose expectation value is to be minimized.
            pool (List[Observable]): the pool of operators to compute the commutator with
            
        Returns:
            list: List of Observable
        """

        return [observable | op for op in tqdm(pool, desc="Computing commutators...")]

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

        # Initialize bound circuit for gradient evaluation
        bound_circuit = copy.copy(circuit)

        # Initialize Result container
        result = Result()

        # Initialize registers
        energy_trace = [] # final energies of each optimization
        operator_idx = []
        optimization_trace = [] # full ansatz optimization traces for each adapt step
        n_iters_optim = [] # number of optimization steps for each current circuit
        
        # Compute commutators
        if self.commutators is None:
            self.commutators = self._compute_commutators(job.observable, self.pool)
            
        pbar = tqdm(range(self.n_iterations))
        # Iterate over number of input number of iterations
        for iter_num in pbar:

            # Step energy register
            energy_gradients = []

            # Loop over operator pool and find the one with biggest energy gradient
            pbar.set_description("Computing energy gradients...")
            for commutator in self.commutators:
                val = -1j*self.execute(bound_circuit.to_job(observable=commutator)).value
                energy_gradients.append(val)
            
            grad_vec_norm = np.linalg.norm(energy_gradients) 
            
            if grad_vec_norm < self.tol_vanishing_grad:
                warnings.warn(
                        "Norm of the energy gradient is below the set threshold. Ending calculation.", stacklevel=2
                    )
                op_ind = None

            # Else pick the one which changes energy the most (if 2 or more are equal, pick the first one)
            else:
                op_ind = np.argmax(np.abs(energy_gradients))

            if op_ind is not None:
                operator_idx.append(op_ind)
                # Grow ansatz
                pbar.set_description("Growing ansatz...")

                current_ansatz = self._grow_ansatz(self.pool[op_ind], iter_num)
                circuit += current_ansatz.to_circ()

                # Optimize the parameters (the job's circuit was updated)
                result = self.execute(job)

                # Update bound circuit
                bound_circuit = circuit(**result.parameter_map)

                # Store optimization results
                energy_trace.append(result.value) # the optimal energy
                n_iters_optim.append(len(eval(result.meta_data["optimization_trace"])))
                optimization_trace += eval(result.meta_data["optimization_trace"]) # the whole trace
                
            # pylint: disable=eval-used
            if not len(operator_idx): # empty list, meaning the circuit was never grown
                warnings.warn(
                        "Optimizing the initial circuit.", stacklevel=2
                    )
                # Optimize the parameters of the untouched circuit
                result = self.execute(job)
                # Store optimization results
                energy_trace.append(result.value) # the optimal energy
                if len(job.circuit.var_dic): # initial circuit is indeed variational
                    n_iters_optim.append(len(eval(result.meta_data["optimization_trace"])))
                    optimization_trace += eval(result.meta_data["optimization_trace"]) # the whole trace
                break
               
            if op_ind is None:
                break

        result.meta_data = {}
        result.meta_data["operator_order"] = str(operator_idx)
        result.meta_data["energy_trace"] = str(energy_trace)
        result.meta_data["optimization_trace"] = str(optimization_trace)
        result.meta_data["n_iters_optim"] = str(n_iters_optim)
        
        return result
