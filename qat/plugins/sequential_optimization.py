import numpy as np
import math
import copy

from qat.plugins.optimizer import Optimizer
from qat.comm.exceptions.ttypes import PluginException


def rescale_params(params, coeff):
    """
    Apply a multiplicative coefficient to all the numbers in a list;

    Args:
        params (list): a list of parameters
        coeff (float): the coefficient to multiply the parameters with
    """

    rescaled_params = {}

    for key, value in params.items():
        try:
            rescaled_params[key] = value * coeff
        except:
            raise PluginException("Parameters must be real-valued!")

    return rescaled_params


class SeqOptimResult:
    """
    Very simple class made so that sequential optimization results have the same API as
    scipy optimization results.

    Args:
        x (list): list of parameters
        fun (float): associated cost function
    """

    def __init__(self, x, fun):
        self.x = x
        self.fun = fun


class SeqOptim(Optimizer):
    """
    This plugin implements the sequential parameter optimization technique (also known as *rotosolve*) described in:

    - `Nakanishi et al. <http://dx.doi.org/10.1103/PhysRevResearch.2.043158>`_ (arXiv:1903.12166) [2020]
    - `Ostaszewski et al. <http://dx.doi.org/10.22331/q-2021-01-28-391>`_ (arXiv:1905.09692) [2021]

    It is a particularization of the :class:`~qat.plugins.Optimizer` class.

    It consists in tuning the parameters of a variational ansatz one after the other, cycling
    several times through all of them, leveraging the parameter shift rule to find a local minimum
    with three measurements of the cost function.

    Such a method can be used only if all parametrized gates are of the form :math:`\exp(-ic\\theta P/2)` where :math:`P`
    is a tensor product of Pauli matrices and :math:`c` a number that must be entered in the 'coeff'
    field of the plugin (e.g. for rotation matrices :math:`RX`, :math:`RY`, :math:`RZ`, :math:`c=1`).

    In the current implementation of the plugin, all parameterized gates are assumed to be of the same type (i.e. have the same
    coefficient :math:`c`).

    .. note::

        The applicability of the method is not checked for when the batch is received.
        It belongs to the user to provide a circuit matching the requirements mentioned above.

    Args:
        ncycles (int, optional): number of times the plugin cycles through each angle, defaults to 10.
            The value to which it should be set
            so that the cost function converges is however strongly problem-dependent.
        coeff (float, optional): rescaling parameter :math:`c` for all the circuit's angles, defaults to 1.
        x0 (np.array, optional): initial value of the parameters. Defaults to None,
            in which case we assume random initialization.
        verbose (bool): whether we want to print intermediary cost function values, defaults to False.
    """

    def __init__(self, ncycles=10, coeff=1, x0=None, verbose=False):
        self.ncycles_roto = ncycles
        self.coeff = coeff
        self.x0 = x0
        self.verbose = verbose
        super(SeqOptim, self).__init__(collective=False)

    def evaluate_aux(self, params):

        rescaled_params = rescale_params(params, self.coeff)

        return self.evaluate(rescaled_params)

    def optimize(self, var_names):

        n_params = len(var_names)

        if self.x0 is not None:
            x0 = list(self.x0)
        else:
            x0 = list(np.random.random(n_params))

        params = {var_names[i]: val for i, val in enumerate(x0)}
        cf = self.evaluate_aux(params)

        for k in range(self.ncycles_roto):

            if self.verbose:
                print("-------Optimization cycle #%i-----------" % (k + 1))

            for d in range(n_params):

                key = str(var_names[d])
                params_plus = copy.deepcopy(params)
                params_plus[key] = params[key] + math.pi / 2
                cf_plus = np.real(self.evaluate_aux(params_plus))

                params_minus = copy.deepcopy(params)
                params_minus[key] = params[key] - math.pi / 2
                cf_minus = np.real(self.evaluate_aux(params_minus))

                params_new = copy.deepcopy(params)
                params_new[key] = (
                    params[key]
                    - math.pi / 2
                    - math.atan2(2 * cf - cf_plus - cf_minus, cf_plus - cf_minus)
                )
                cf_new = np.real(self.evaluate_aux(params_new))

                # update
                params = params_new
                cf = cf_new

            if self.verbose:
                print("current cost:", cf)

        result = SeqOptimResult(params.values(), cf)

        return result.fun, list(result.x), result