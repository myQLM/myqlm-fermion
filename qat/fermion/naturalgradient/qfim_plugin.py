# -*- coding: utf-8 -*-
"""
Gradient Descend Plugin including natural gradient descent
"""

from typing import List
from itertools import product
import numpy as np

from qat.comm.exceptions.ttypes import PluginException
from qat.core.optimizer import Optimizer

from ..matchgates import gate_set
from .auto_derivatives import (
    auto_differentiation_gradient_dictionary,
    auto_differentiation_qfim_dictionaries,
)


class GradientDescentOptimizer(Optimizer):
    r"""Gradient-based optimization plugin, with possibility to use natural gradients
    as described in `this publication <http://dx.doi.org/10.22331/q-2020-05-25-269>`_.

    The variational parameters are updated according to the rule

    .. math::

        \vec{\theta}_{k+1} = \vec{\theta}_{k} - \eta g^{+} \vec{\nabla} E

    with :math:`E(\vec{\theta}) = \langle \psi(\vec{\theta}) | H | \psi (\vec{\theta}) \rangle`,
    :math:`\left [ \vec{\nabla} E \right ]_i = \frac{\partial E}{\partial \theta_i}` and

    the metric tensor

    .. math::

        g_{ij} = \mathrm{Re} \left[ \left \langle \frac{\partial \psi}{\partial \theta_i} \Bigg | \frac{\partial \psi}{\partial \theta_j} \right \rangle
        - \left \langle \frac{\partial \psi}{\partial \theta_i} \Bigg |  \psi \right \rangle
        \left \langle  \psi \Bigg | \frac{\partial \psi}{\partial \theta_j} \right \rangle     \right ]

    For regular gradient descent, we choose :math:`g = I`.

    Args:
        maxiter (int, optional): Maximum number of iterations.
            Defaults to 1000.
        natural_gradient (bool, optional) : Whether to perform natural gradient descent or "classical"
            gradient descent. Default to True.
        lambda_step (float, optional): Gradient descent step size :math:`\lambda`.
            Defaults to 0.2.
        stop_crit (string, optional): Stopping criterion, among {"grad_norm"|"energy_dist"}.
            Defaults to grad_norm.
        tol (float, optional): Tolerance for stopping criterion.
            Defaults to 1e-10.
        x0 (list, optional): Initial value of the parameters. The indexing must be the same as for the variables obtained via the
            `.get_variables()` method. If None, the initial parameters will be randomly chosen. Defaults to None.

    """

    def __init__(
        self,
        maxiter: int = 200,
        lambda_step: float = 0.2,
        natural_gradient: bool = True,
        stop_crit: str = "grad_norm",
        tol: float = 1e-3,
        x0: List[float] = None,
        user_custom_gates=None,
        collective: bool = False,
    ):

        self.lambda_step = lambda_step
        self.natural_gradient = natural_gradient

        # Parameters values
        self.parameters_index = {}
        self.initial_parameters = x0
        self.parameters_dict = None

        # For stopping criterion purposes
        self.maxiter = maxiter  # Maximum number of iterations
        self.stop_crit = stop_crit
        self.tolerance = tol  # For stopping criterion
        self.check_crit_val = 1.0 + self.tolerance
        self.iterations = 0

        # For custom gates (eg. XX)
        self.custom_gates = user_custom_gates
        self.my_gate_set = gate_set

        # TODO: Implement collective=True
        if collective:
            raise NotImplementedError("GradientDescentOptimizer is currently only compatible with collective=False.")

        super().__init__(collective=collective)

    def execute_weighted_jobs_list(self, jobs_list):
        """
        Auxiliary function to execute the jobs from the given list.
        """
        res_weighted = []
        for coeff, my_test_job in jobs_list:

            temp_job = my_test_job(gate_set=self.my_gate_set, **self.parameters_dict)

            res = self.execute(temp_job)
            res_weighted.append(coeff * res.value)

        return res_weighted

    def optimize(self, var_names):
        """
        The main method for the Junction plugin.
        """

        if self.initial_parameters is None:

            # By default, initiate search from (0., 0., ..., 0.)
            self.initial_parameters = np.random.uniform(0, 2 * np.pi, len(var_names))

        self.parameters_dict = dict(zip(var_names, self.initial_parameters))

        # Begin search from the given values if the dict keys are the same (it has already been copied)
        if list(self.parameters_dict) != var_names:

            raise PluginException(
                message=f"Initial parameters dict keys ({list(self.parameters_dict.keys())}) do not match the job parameters"
                "names. Program parameters are : {parameters}"
            )

        # Keep track of which parameter corresponds to which index in the gradient, matrix, ... (assumes dictionaries are ordered)
        for ind, varkey in enumerate(self.parameters_dict.keys()):
            self.parameters_index[ind] = varkey

        nb_parameters = len(self.parameters_dict)

        angles = []

        angles.append([val for val in self.parameters_dict.values()])  # Assumes dictionaries are ordered

        pristine_job = self.job(gate_set=self.my_gate_set, **self.parameters_dict)

        energy = self.execute(pristine_job).value
        self.trace.append(energy)

        gradient_jobs_dict = auto_differentiation_gradient_dictionary(
            self.job, self.job.observable, self.parameters_dict, user_custom_gates=self.custom_gates
        )

        if self.natural_gradient:
            (qfim_dkdl_jobs_dict, qfim_dkpsi_jobs_dict, qfim_psidl_jobs_dict) = auto_differentiation_qfim_dictionaries(
                self.job, nb_parameters, self.parameters_index, user_custom_gates=self.custom_gates
            )

        self.iterations = 0
        stopping_criterion = False

        while self.iterations < self.maxiter and not stopping_criterion:

            q_gradient = {var_key: 0.0 for var_key in self.parameters_dict}

            for var_key in self.parameters_dict:

                jobs_list = gradient_jobs_dict[var_key]

                val_list = self.execute_weighted_jobs_list(jobs_list)
                q_gradient[var_key] = sum(val_list)

            # Correct gradient with the Quantum Fisher Information Matrix as a metric tensor
            if self.natural_gradient:

                dkpsi, psidl = [], []
                for k in range(nb_parameters):

                    # Compute <dk|psi>
                    jobs_list = qfim_dkpsi_jobs_dict[self.parameters_index[k]]
                    val_list = self.execute_weighted_jobs_list(jobs_list)
                    dkpsi.append(sum(val_list))  # should be a pure imaginary number

                    # Compute <psi|dl>
                    jobs_list = qfim_psidl_jobs_dict[self.parameters_index[k]]
                    val_list = self.execute_weighted_jobs_list(jobs_list)
                    psidl.append(sum(val_list))  # should be a pure imaginary number

                qfim = np.zeros((nb_parameters, nb_parameters), dtype=float)
                for k, l in product(range(nb_parameters), repeat=2):

                    # Compute <dk|dl>
                    jobs_list = qfim_dkdl_jobs_dict[self.parameters_index[k] + ";" + self.parameters_index[l]]
                    val_list = self.execute_weighted_jobs_list(jobs_list)
                    dkdl = sum(val_list)

                    qfim[k][l] = 4 * (dkdl - dkpsi[k] * psidl[l])

                # Truncate zero values
                # qfim = np.where(np.abs(qfim) < 1e-10, 0.0, qfim)
                qfim[np.abs(qfim) < 1e-10] = 0

                # Inverse the metric tensor
                invqfim = np.linalg.pinv(qfim)

            # Perform optimization step (update parameters)
            # $\vec{\vartheta} <- \vec{\vartheta} - \lambda_step * [qfim]^{-1} * \nabla E$

            cur_params = np.array([p_val for p_val in self.parameters_dict.values()])
            cur_grad = np.array([grad_i for grad_i in q_gradient.values()])

            # Truncate zero values
            cur_grad = np.real(np.where(np.abs(cur_grad) < 1e-10, 0.0, cur_grad))

            if self.natural_gradient:

                try:
                    soldiff = np.linalg.solve(qfim, -self.lambda_step * cur_grad)
                    cur_params = np.real(cur_params + soldiff)

                except np.linalg.LinAlgError:
                    cur_params = np.real(cur_params - self.lambda_step * invqfim.dot(cur_grad))

            else:
                cur_params = np.real(cur_params - self.lambda_step * cur_grad)

            for idx, param in self.parameters_index.items():
                self.parameters_dict[param] = cur_params[idx]

            # Update Optimization Data
            angles.append([ang for ang in self.parameters_dict.values()])  # Assumes dictionaries are ordered

            # Temporary for XX gate
            binded_job = self.job(gate_set=self.my_gate_set, **self.parameters_dict)

            energy = self.execute(binded_job).value

            if self.stop_crit == "energy_dist" and self.iterations > 0:
                self.check_crit_val = np.abs(energy - self.trace[-1])

            self.trace.append(energy)

            if self.stop_crit == "grad_norm":
                self.check_crit_val = np.linalg.norm(cur_grad)

            if self.stop_crit != "none":

                if self.check_crit_val < self.tolerance:
                    stopping_criterion = True

            self.iterations += 1

        # Generate (angle, energy) tuple for optimizer trace metadata
        angle_energy = [(angles[idx], self.trace[idx]) for idx in range(len(self.trace))]

        return (self.trace[-1], self.parameters_dict.values(), {"iterations": self.iterations, "energies": angle_energy})
