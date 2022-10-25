# -*- coding: utf-8 -*-
"""
Zero Noise (linear or exponential) Extrapolation plugin.
"""

from typing import List, Optional
import numpy as np
from scipy.stats import linregress
import copy

from qat.core import BatchResult, Circuit, Batch
from qat.core.plugins import AbstractPlugin
from qat.lang.AQASM import CNOT
from qat.lang.AQASM.gates import Gate
from qat.comm.exceptions.ttypes import PluginException


def insert_ids(circ: Circuit, gates: list, n_ins: int) -> Circuit:
    r"""
    Insert a number n_ins of GG^{\dagger} after each occurence of gate G in the circuit.
    In the absence of noise,  GG^{\dagger}=id is without effect.

    Args:
        circ (Circuit): Circuit.
        gates (list): The gates G to duplicate.
        n_ins (int): The number of decompositions GG^{\dagger} of the identity to insert after each occurence of G in circ.

    Returns:
        modified_circ (Circuit): The initial circuit, with GG^{\dagger} insertions.

    """

    modified_circ = copy.deepcopy(circ)

    n_gates_inserted = 0
    op_index = 0

    for op in circ.iterate_simple():

        for gate in gates:

            if op[0] == gate.name:

                for _ in range(n_ins):
                    modified_circ.insert_gate(gate=gate, position=op_index + n_gates_inserted + 1, qbits=op[2])
                    modified_circ.insert_gate(
                        gate=gate.dag(),
                        position=op_index + n_gates_inserted + 2,
                        qbits=op[2],
                    )
                n_gates_inserted += 2 * n_ins

        op_index += 1

    return modified_circ


def extract_values(batch_result: BatchResult, n_ins: int, n_jobs: int, job_number: int) -> list:
    r"""
    Given a batch result corresponding to

    :math:`C_0^1, C_0^2, ..., C_0^{n_\mathrm{jobs}}, C_1^1, C_1^2, ..., C_1^{n_\mathrm{jobs}}, ..., C_{n_\mathrm{ins}}^{n_\mathrm{jobs}}`

    where :math:`C_i^j` is the circuit for job :math:`j` in which :math:`i` :math:`GG^{\dagger}` were inserted after each :math:`G`,
    this function returns the list of (n_ins + 1) values that are used to compute
    the extrapolated value for the job job_number.

    Args:
        batch_result (BatchResult): A batch result containing results for all jobs and points.
        n_ins (int): The maximal number of GG^{\dagger} inserted.
        n_jobs (int): The number of jobs that were initially sent to the stack.
        job_number (int): The index of the job we want to isolate the meaningful result values for.

    Returns:
        values_for_fit (list): List of measured values to use to perform the extrapolation.

    """

    results = batch_result.results  # list of Result objects

    values_for_fit = []

    # first append the result corresponding to
    values_for_fit.append(results[job_number].value)

    # the circuit without id insertions
    for i in range(n_ins):
        index = n_jobs + job_number * n_ins + i
        values_for_fit.append(results[index].value)

    return values_for_fit


def perform_extrapolation(
    values_for_fit: list,
    n_ins: int,
    extrap_method: Optional[str] = "linear",
    asymptot: Optional[float] = 0,
):
    r"""
    Perform an extrapolation to zero noise.

    Args:
        values_for_fit (list): Values to carry the fit on.
        n_ins (int): The maximal number of GG^{\dagger} insertions.
        extrap_method (Optional[str]): Which kind of extrapolation to make, defaults to 'linear'. The other choice is 'exponential'.
        asymptot (Optional[float]): Asymptotic value of the observable as n_ins goes to infinity. Must be known for exponential
            extrapolation. Defaults to 0.

    Returns:
        value (float): The zero-noise extrapolated value

    """

    try:
        assert len(values_for_fit) == n_ins + 1

    except Exception as exc:
        raise PluginException(
            "Not enough jobs in the batch ({len(values_for_fit)}) compared with the max number"
            "of local CNOT insertions ({n_ins}) extrapolation"
        ) from exc

    if extrap_method == "linear" or not all([val - asymptot != 0 for val in values_for_fit]):
        a, b, _, _, _ = linregress(range(n_ins + 1), values_for_fit)
        value = -0.5 * a + b

    elif extrap_method == "exponential":

        sign = np.sign(values_for_fit[0] - asymptot)
        log_values = [np.log(abs(val - asymptot)) for val in values_for_fit]
        a, b, _, _, _ = linregress(range(n_ins + 1), log_values)
        value = sign * np.exp(-0.5 * a + b) + asymptot

    else:
        raise PluginException(
            f"Extrapolation method not implemented. You required the {extrap_method} method whereas only linear and exponential options exist.",
        )

    return value, a, b


class ZeroNoiseExtrapolator(AbstractPlugin):
    r"""
    Perform Zero-Noise Extrapolation (linear - by default - or exponential) by inserting
    decompositions :math:`GG^{\dagger}` of the identity after each occurrence of the gate
    :math:`G` (default is CNOT) in the circuit.

    More specifically, let

    .. math::
            f(n_\mathrm{pairs}) = \mathrm{Tr} \left[\rho_{n_\mathrm{pairs}} \hat{O} \right]

    be the noisy expectation value of the observable of interest :math:`\hat{O}`
    on the circuit in which each gate :math:`G` was replaced by :math:`G(GG^{\dagger})^{n_{\mathrm{pairs}}}`.
    In other words, this value corresponds to a noise strength :math:`\mathcal{N} \propto 1 + 2 n_\mathrm{pairs}`.

    Then the plugin will fit :math:`f` from the :math:`(n_{\mathrm{ins}}+1)` points corresponding to
    :math:`n_{\mathrm{pairs}} =0...n_{\mathrm{ins}}` with either a linear ansatz

    - :math:`f(n_\mathrm{pairs}) = an_\mathrm{pairs} + b`

    or an exponential one

    - :math:`f(n_\mathrm{pairs}) = \pm e^{an_\mathrm{pairs}+b} + C`

    and infer a noise-free value for :math:`\langle \hat{O}_{\mathrm{noise-free, inferred}} \rangle` as :math:`f(-0.5)`.

    In the latter case, :math:`C` corresponds to the observable's expectation value over the totally mixed state,
    which can be shown to be the observable's constant coefficient and is the asymptotic value reached as the noise
    increases, whereas the sign is case-specific and is evaluated by the plugin by looking at the difference
    between the noisy value and the asymptot :math:`C`).

    .. note::
        Be careful as the plugin will not work correctly if the batch sent to the stack
        mixes sampling and observable measurement jobs!

    Args:
        n_ins (Optional[int]): Maximum number of identity insertions to go to. Defaults to 1.
        extrap_gates (Optional[List[Gate]]): Gates :math:`G` to be followed by identity decompositions :math:`GG^{\dagger}`.
            Defaults to CNOT
        extrap_method (Optional[str]): Form of the ansatz fit, defaults to 'linear'. Can be also 'exponential'.

    """

    def __init__(
        self,
        n_ins: Optional[int] = 1,
        extrap_gates: Optional[List[Gate]] = None,
        extrap_method: Optional[str] = "linear",
    ):
        if extrap_gates is None:
            extrap_gates = [CNOT]

        super(ZeroNoiseExtrapolator, self).__init__()
        self.n_ins = n_ins
        self.extrap_gates = extrap_gates
        self.extrap_method = extrap_method

        self.is_sampling = None
        self.asymptots = None

    def compile(self, batch: Batch, specs):
        """
        Compile the batch.

        Args:
            batch (:class:`~qat.core.Batch`): Batch to optimize

        """

        # Initialize resulting batch
        resulting_batch = copy.deepcopy(batch)

        if batch.jobs and not hasattr(batch.jobs[0].observable, "constant_coeff"):
            self.is_sampling = True
            return resulting_batch

        self.is_sampling = False

        if self.extrap_method == "exponential":
            self.asymptots = []

        elif self.extrap_method == "linear":
            self.asymptots = [0] * len(batch.jobs)

        for _, job in enumerate(batch.jobs):

            for i in range(self.n_ins):

                # Duplicate each job, changing the circuit to a circuit with GG^{\dagger} insertions
                modified_job = copy.deepcopy(job)
                circ = job.circuit
                modified_circ = insert_ids(circ, self.extrap_gates, i + 1)
                modified_job.circuit = modified_circ
                resulting_batch.jobs.append(modified_job)

            if self.extrap_method == "exponential":
                # Fetch the expected asymptotic value
                asymptot = job.observable.constant_coeff
                self.asymptots.append(asymptot)

            else:
                self.asymptots.append(0)

        # Return batch to send to the QPU
        return resulting_batch

    def post_process(self, batch_result: BatchResult) -> BatchResult:
        """
        Perform post processing.

        Args:
            batch_result (:class:`~qat.core.BatchResult`): Result to post process.

        Returns:
            :class:`~qat.core.BatchResult`

        """

        if self.is_sampling:
            return batch_result

        number_of_jobs_involved = self.n_ins + 1
        n_jobs_init = len(batch_result) // (number_of_jobs_involved)

        extrapolated_results = BatchResult(meta_data=batch_result.meta_data)
        for i in range(n_jobs_init):

            result_to_fix = batch_result.results[i]
            extrapolated_results.results.append(result_to_fix)

            if result_to_fix.value is not None:

                values_for_fit = extract_values(batch_result, self.n_ins, n_jobs_init, i)
                result_to_fix.meta_data["values_for_ZNE"] = str(values_for_fit)
                extrapolated_results.results[i].value, a, b = perform_extrapolation(
                    values_for_fit,
                    self.n_ins,
                    extrap_method=self.extrap_method,
                    asymptot=self.asymptots[i],
                )

                result_to_fix.meta_data["ZNE_fit_parameters"] = str({"a": a, "b": b})

        return extrapolated_results
