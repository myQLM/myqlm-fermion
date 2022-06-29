import numpy as np
import time
from qat.comm.exceptions.ttypes import PluginException
from qat.core import Result, Job
from qat.plugins.junction import Junction

from .auto_derivatives import (
    auto_differentiation_gradient_dictionary,
    auto_differentiation_QFIM_dictionaries,
)

# Temporary to handle XX gate
from qat.fermion.matchgates import add_gates_to_default_gate_set


class GradientMinimizePlugin(Junction):
    r"""Gradient-based optimization plugin, with possibility to use natural gradients
    as described in `this publication <http://dx.doi.org/10.22331/q-2020-05-25-269>`_.

    The variational parameters are updated according to the rule

    .. math::

        \vec{\theta}_{k+1} = \vec{\theta}_{k} - \lambda g^{-1} \vec{\nabla} E

    with :math:`E(\vec{\theta}) = \langle \psi(\vec{\theta}) | H | \psi (\vec{\theta}) \rangle`,
    :math:`\left [ \vec{\nabla} E \right ]_i = \frac{\partial E}{\partial \theta_i}` and

    the metric tensor

    .. math::

        g_{ij} = \mathrm{Re} \left[ \left \langle \frac{\partial \psi}{\partial \theta_i} \Bigg | \frac{\partial \psi}{\partial \theta_j} \right \rangle
        - \left \langle \frac{\partial \psi}{\partial \theta_i} \Bigg |  \psi \right \rangle
        \left \langle  \psi \Bigg | \frac{\partial \psi}{\partial \theta_j} \right \rangle     \right ]

    For regular gradient descent, we choose :math:`g = I`.

    Args:
        maxiter (int, optional): maximum number of iterations.
            Defaults to 1000.
        natural_gradient (bool, optional) : Whether to perform natural gradient descent or "classical"
            gradient descent. Default to True.
        lambda_step (float, optional): Gradient descent step size :math:`\lambda`.
            Defaults to 0.2.
        stop_crit (string, optional): stopping criterion, among {"grad_norm"|"energy_dist"}.
            Defaults to grad_norm.
        tol (float, optional): tolerance for stopping criterion.
            Defaults to 1e-10.
        x0 (dict, optional): initial value of the parameters. If None, the initial parameters will be randomly chosen. Defaults to
            None.

    """

    def __init__(
        self,
        maxiter=1000,
        lambda_step=0.2,
        natural_gradient=True,
        do_compute_energy=True,
        stop_crit="grad_norm",
        tol=1e-10,
        x0=None,
        user_custom_gates=None,
    ):

        if do_compute_energy == False and stop_crit == "energy_dist":
            raise PluginException("Can't check energy update steps if 'do_compute_enregy' is set to %s" % do_compute_energy)

        self.lambda_step = lambda_step
        self.natural_gradient = natural_gradient
        self.do_compute_energy = do_compute_energy

        # Parameters values
        self.parameters_index = {}
        self.parameters_values = x0.copy() if x0 is not None else None  # Dict for the values of the circuit parameters

        # For stopping criterion purposes
        self.maxiter = maxiter  # Maximum number of iterations
        self.stop_crit = stop_crit
        self.tolerance = tol  # For stopping criterion
        self.check_crit_val = 1.0 + self.tolerance
        self.iterations = 0

        # Plugin history after instantiation
        self.nb_use = 0

        # For custom gates (eg. XX)
        self.custom_gates = user_custom_gates
        self.my_gate_set = add_gates_to_default_gate_set()

        super(GradientMinimizePlugin, self).__init__(collective=False)

    def execute_weighted_jobs_list(self, jobs_list):
        """
        Auxiliary function to execute the jobs from the given list.
        """
        res_weighted = []
        for coeff, my_test_job in jobs_list:

            temp_job = my_test_job(gate_set=self.my_gate_set, **self.parameters_values)

            res = self.execute(temp_job)
            val = res.value
            res_weighted.append(coeff * val)

        return res_weighted

    def run(self, qlm_object: Job, _) -> Result:
        """
        The main method for the Junction plugin.
        """

        start_time = time.time()

        # The job and its hamiltonian observable
        job = qlm_object()
        hamilt = job.observable

        # Initial values to be explored
        parameters = qlm_object.get_variables()

        if self.parameters_values is None:

            # By default, initiate search from (0., 0., ..., 0.)
            self.parameters_values = {
                variable: value for variable, value in zip(parameters, np.random.uniform(0, 2 * np.pi, len(parameters)))
            }

        else:

            # Begin search from the given values if the dict keys are the same (it has already been copied)
            if not (list(self.parameters_values.keys()) == parameters):

                raise PluginException(
                    message="Initial parameters dict keys (%s) do not match the job parameters names. Program parameters are : %s"
                    % (list(self.parameters_values.keys()), parameters)
                )

        # Keep track of which parameter corresponds to which index in the gradient, matrix, ... (assumes dictionaries are ordered)
        # for ind, varkey in enumerate(parameters): # -> pb with the indices ?
        for ind, varkey in enumerate(self.parameters_values.keys()):
            self.parameters_index[ind] = varkey

        nb_parameters = len(self.parameters_values)

        energy_trace = None

        if self.do_compute_energy:
            energy_trace = []

        angle_trace = []

        angle_trace.append([val for val in self.parameters_values.values()])  # Assumes dictionaries are ordered...

        if self.do_compute_energy:

            pristine_job = qlm_object(gate_set=self.my_gate_set, **self.parameters_values)

            energy = self.execute(pristine_job).value
            energy_trace.append(energy)

        gradient_jobs_dict = auto_differentiation_gradient_dictionary(
            job, hamilt, self.parameters_values, user_custom_gates=self.custom_gates
        )

        if self.natural_gradient:
            (QFIM_dkdl_jobs_dict, QFIM_dkpsi_jobs_dict, QFIM_psidl_jobs_dict) = auto_differentiation_QFIM_dictionaries(
                job, nb_parameters, self.parameters_index, user_custom_gates=self.custom_gates
            )

        self.iterations = 0
        stopping_criterion = False

        while self.iterations < self.maxiter and stopping_criterion == False:

            q_gradient = {var_key: 0.0 for var_key in self.parameters_values}

            for var_key in self.parameters_values:

                jobs_list = gradient_jobs_dict[var_key]

                val_list = self.execute_weighted_jobs_list(jobs_list)
                q_gradient[var_key] = sum(val_list)

            if self.natural_gradient:

                # The plugin was asked to correct gradient with the Quantum Fisher Information Matrix as a metric tensor
                QFIM = np.zeros((nb_parameters, nb_parameters), dtype=float)

                for k in range(nb_parameters):

                    # Compute <dk|psi>
                    jobs_list = QFIM_dkpsi_jobs_dict[self.parameters_index[k]]
                    val_list = self.execute_weighted_jobs_list(jobs_list)
                    dkpsi = sum(val_list)  # should be a pure imaginary number

                    for l in range(nb_parameters):

                        # Compute <dk|dl>
                        jobs_list = QFIM_dkdl_jobs_dict[self.parameters_index[k] + ";" + self.parameters_index[l]]
                        val_list = self.execute_weighted_jobs_list(jobs_list)
                        dkdl = sum(val_list)

                        # Compute <psi|dl>
                        jobs_list = QFIM_psidl_jobs_dict[self.parameters_index[l]]
                        val_list = self.execute_weighted_jobs_list(jobs_list)
                        psidl = sum(val_list)  # should be a pure imaginary number

                        QFIM[k][l] = 4 * (dkdl - dkpsi * psidl)

                # Get rid of too small values
                QFIM = np.where(np.abs(QFIM) < 1e-10, 0.0, QFIM)

                # Inverse the metric tensor
                invQFIM = np.linalg.pinv(QFIM)

            # Perform optimisation step (update parameters)
            # $\vec{\vartheta} <- \vec{\vartheta} - \lambda_step * [QFIM]^{-1} * \nabla E$

            cur_params = np.array([p_val for p_val in self.parameters_values.values()])
            cur_grad = np.array([grad_i for grad_i in q_gradient.values()])

            # Get rid of too small values
            cur_grad = np.real(np.where(np.abs(cur_grad) < 1e-10, 0.0, cur_grad))

            if self.natural_gradient:

                try:
                    soldiff = np.linalg.solve(QFIM, -self.lambda_step * cur_grad)
                    cur_params = np.real(cur_params + soldiff)

                except np.linalg.LinAlgError:
                    cur_params = np.real(cur_params - self.lambda_step * invQFIM.dot(cur_grad))

            else:
                cur_params = np.real(cur_params - self.lambda_step * cur_grad)

            for ind_param in self.parameters_index:
                self.parameters_values[self.parameters_index[ind_param]] = cur_params[ind_param]

            # Update Optimisation Data
            angle_trace.append([ang for ang in self.parameters_values.values()])  # Assumes dictionaries are ordered...

            if self.do_compute_energy:

                # Temporary for XX gate
                binded_job = qlm_object(gate_set=self.my_gate_set, **self.parameters_values)

                energy = self.execute(binded_job).value

                if self.stop_crit == "energy_dist" and self.iterations > 0:
                    self.check_crit_val = np.abs(energy - energy_trace[-1])

                energy_trace.append(energy)

            if self.stop_crit == "grad_norm":
                self.check_crit_val = np.linalg.norm(cur_grad)

            if not (self.stop_crit == "none"):

                if self.check_crit_val < self.tolerance:
                    stopping_criterion = True

            self.iterations += 1

        end_time = time.time()

        # Return Result with some meta data
        self.nb_use += 1  # The plugin has successfully run

        if self.natural_gradient:
            title = r"Natural Gradient descent with QFIM. $\lambda =$ %f, nb_calls = %i" % (self.lambda_step, self.iterations)
        else:
            title = r"Gradient descent without QFIM. $\lambda =$ %f, nb_calls = %i" % (self.lambda_step, self.iterations)

        return Result(
            value=energy_trace[-1],  # supposing last energy is the minimum
            meta_data={
                "title": title,
                "elapsed_time": str(end_time - start_time),
                "nb_iterations": str(self.iterations),
                "parameters": str(self.parameters_values),
                "optimization_trace": str(energy_trace),
                "parameters_index": str(self.parameters_index),
                "parameters_trace": str(angle_trace),
            },
        )
