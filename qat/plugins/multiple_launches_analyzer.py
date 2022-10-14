# -*- coding: utf-8 -*-
"""
MultipleLaunchesAnalyzer plugin
"""

from typing import Optional
import numpy as np
import copy

from qat.core import BatchResult
from qat.core.plugins import AbstractPlugin


class MultipleLaunchesAnalyzer(AbstractPlugin):
    """
    A very simple plugin that duplicates each job of the batch
    and for each keeps the result associated to the lowest observed value.

    The plugin may be used to carry out the batch optimization
    of the angles of a fixed ansatz starting from different random
    initializations. To this end, the stack upstream must include a
    VQE optimizer instantiated without any starting point.

    Args:
        n_runs (Optional[int]): Number of initializations investigated. Defaults to 5.
        verbose (Optional[bool]): Print infos. Defaults to False.
    """

    def __init__(self, n_runs: Optional[int] = 5, verbose: Optional[bool] = False):
        super(MultipleLaunchesAnalyzer, self).__init__()
        self.n_runs = n_runs
        self.verbose = verbose

    def compile(self, batch, specs):
        """
        Compile the batch.

        Args:
            batch (:class:`~qat.core.Batch`): Batch to optimize.
        """

        # Initialize resulting batch
        resulting_batch = copy.deepcopy(batch)
        resulting_batch.jobs = []  # empty job list

        for _, job in enumerate(batch.jobs):

            for _ in range(self.n_runs):

                # Duplicate each job, so that several random initializations are trialled
                job_copy = copy.deepcopy(job)
                resulting_batch.jobs.append(job_copy)

        # Return batch to send to the qpu
        return resulting_batch

    def post_process(self, batch_result: BatchResult) -> BatchResult:
        """
        Perform post processing.

        Args:
            batch_result (:class:`~qat.core.BatchResult`): Result to post process.

        Returns:
            :class:`~qat.core.BatchResult`

        """

        n_jobs_init = len(batch_result) // (self.n_runs)

        best_results = BatchResult()

        vals = np.reshape(
            np.array([result.value for result in batch_result.results]),
            (n_jobs_init, self.n_runs),
        )

        for i in range(n_jobs_init):

            # Select row corresponding to job
            vals_for_job = vals[i, :]

            if self.verbose:
                print("Values found for job #%i" % (i + 1), vals_for_job)
                best_val = np.min(vals_for_job)
                print("Lowest value is thus", best_val)

            local_index = np.argmin(vals_for_job)
            pos_in_batch = i * self.n_runs + local_index

            if self.verbose:
                print("Position in batch of job with lowest value:", pos_in_batch)

            var = np.var(vals_for_job)
            res_to_keep = batch_result[pos_in_batch]
            res_to_keep.meta_data["optimal_values_variance"] = str(var)
            res_to_keep.meta_data["reached_values"] = str(vals_for_job.tolist())
            best_results.results.append(res_to_keep)

        return best_results
