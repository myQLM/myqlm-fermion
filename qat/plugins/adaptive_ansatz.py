import time
import numpy as np
from scipy.optimize import minimize

from qat.lang.AQASM import Program

from qat.core import Result
from qat.plugins import Junction
from qat.comm.exceptions.ttypes import PluginException, ErrorType


class AdaptiveAnsatzPlugin(Junction):

    r"""Adaptive ansatz plugin constructs an ansatz by selecting operators
    from a given pool of operators.

    Args:
        operator_pool (list<Observable>): list of operators \tau_i to appear as gates
            exp(-i theta_i * \tau_i) in parametric circuit
        commutators (list<Observable>, optional): list of commutators [H, \tau_i], with
            H Hamiltonian to be minimized. Defaults to None, in which case the
            commutators are computed by the plugin.
        max_iter (int, optional): maximum number of iterations or operators added to the ansatz

    Notes:
        See 1) https://www.nature.com/articles/s41467-019-10988-2.pdf
            2) http://arxiv.org/abs/1911.10205
    """

    def __init__(
        self,
        operator_pool,
        commutators=None,
        max_iter=10,
        verbose=False,
        use_external_optimizer=False,
        tol=1e-5,
        max_vqe_iter=100,
        method="BFGS",
    ):
        super(AdaptiveAnsatzPlugin, self).__init__()
        self.operator_pool = operator_pool
        self.commutators = commutators
        self.parametric_circuit = None
        self.circuit = None
        self.max_iter = max_iter
        self.verbose = verbose
        self.use_external_optimizer = use_external_optimizer
        self.tol = tol
        self.max_vqe_iter = max_vqe_iter
        self.method = method

    def calculate_commutators(self, obs):
        self.commutators = []
        for op in self.operator_pool:
            self.commutators.append(op | obs)

    def initialize_circuit(self, job):
        if job.circuit is None:
            prog = Program()
            qreg = prog.qalloc(job.observable.nbqbits)
            self.parametric_circuit = prog.to_circ()
            self.circuit = prog.to_circ()
        else:
            self.parametric_circuit = job.circuit
            self.circuit = job.circuit

    def evaluate_gradients(self):
        gradients = []
        for op in self.commutators:
            val = (self.execute(self.circuit.to_job(observable=op))).value
            gradients.append(val)
            if self.verbose:
                if abs(val) > 1e-12:
                    print("Op: %s, <psi|op|psi>= %s" % (op, val))
        return gradients

    def grow_ansatz(self, op_ind, iter_num):
        prog = make_trotter_slice(self.operator_pool[op_ind], iter_num)
        self.parametric_circuit += prog.to_circ()

    def find_angles(self, job, x0):
        trace = []
        n_params = len(job.get_variables())
        assert len(x0) == n_params

        def fun(x):
            circ = job.circuit(**{"theta_" + str(j): elm for j, elm in enumerate(x)})
            _job = circ.to_job(observable=job.observable)
            energy = np.real(self.execute(_job).value)
            trace.append(energy)
            return energy

        res = minimize(
            fun,
            x0,
            method=self.method,
            tol=self.tol,
            options={"maxiter": self.max_vqe_iter},
        )
        return res.fun, list(res.x), trace

    def reset_circuits(self):
        self.circuit = None
        self.parametric_circuit = None

    def do_compatibility_check(self, job):
        if job.observable is None:
            raise PluginException(
                code=ErrorType.ABORT,
                message="An observable should be specified in the job",
            )

        for i, op in enumerate(self.operator_pool):
            if op.nbqbits != job.observable.nbqbits:
                raise PluginException(
                    code=ErrorType.ABORT,
                    message="operator number {0} acts on {1} qubit(s)\
                                      but the observable in the job acts on {2} qubit(s)".format(
                        i, op.nbqbits, job.observable.nbqbits
                    ),
                )

    def run(self, job, meta_data):
        if self.verbose:
            print("Checking compatibility")
        self.do_compatibility_check(job)
        op_indices = []
        energy_trace = []
        theta = []
        result = Result()

        if self.commutators is None:
            if self.verbose:
                print("Computing commutators")
                st = time.time()
            self.calculate_commutators(job.observable)
            if self.verbose:
                print("Done. Took %s secs" % (time.time() - st))

        if self.verbose:
            print("Initializating")
        self.reset_circuits()
        self.initialize_circuit(job)

        if self.verbose:
            print("Starting iterations...")
        for i in range(self.max_iter):
            st = time.time()
            if self.verbose:
                print("Iteration number {0}".format(i))
            gradients = self.evaluate_gradients()
            op_ind = np.argmax(np.abs(gradients))
            op_indices.append(op_ind)
            self.grow_ansatz(op_ind, i)
            theta.append(0.0)

            if self.verbose:
                print("Best operator {0}".format(op_ind))
                print("Gradients :", gradients)

            job = self.parametric_circuit.to_job(observable=job.observable)

            if self.use_external_optimizer:
                result = self.execute(
                    self.parametric_circuit.to_job(observable=job.observable)
                )
                theta = eval(result.meta_data["parameters"])
                energy_trace += eval(result.meta_data["optimization_trace"])
            else:
                result.value, theta, trace = self.find_angles(job, theta)
                energy_trace += trace

            self.circuit = self.parametric_circuit(
                **{"theta_" + str(j): elm for j, elm in enumerate(theta)}
            )
            if self.verbose:
                print("Current energy {0}".format(result.value))
                print("Done. Took %s secs" % (time.time() - st))

        result.meta_data = dict()
        result.meta_data["operator_order"] = str(op_indices)
        result.meta_data["optimization_trace"] = str(energy_trace)
        return result