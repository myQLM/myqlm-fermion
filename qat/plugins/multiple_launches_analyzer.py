import numpy as np
import copy

from qat.core import BatchResult
from qat.core.plugins import AbstractPlugin

from qat.core.console import display


class MultipleLaunchesAnalyzer(AbstractPlugin):
    """
    A very simple plugin that duplicates each job of the batch
    and for each keeps the result associated to the lowest observed value.

    The plugin may be used to carry out the batch optimization
    of the angles of a fixed ansatz starting from different random
    initializations. To this end, the stack upstream must include a
    VQE optimizer instantiated without any starting point.

    Args:
        n_runs (int, optional): number of initializations investigated, default to 5.
        verbose (bool, optional): print infos, defaults to False.
    """

    def __init__(self, n_runs=5, verbose=False):
        super(MultipleLaunchesAnalyzer, self).__init__()
        self.n_runs = n_runs
        self.verbose = verbose

    def compile(self, batch, specs):
        """
        Compile the batch

        Args:
            batch (:class:`~qat.core.Batch`): batch to optimize
        """
        # Initialize resulting batch
        resulting_batch = copy.deepcopy(batch)
        resulting_batch.jobs = []  # empty job list

        for idx, job in enumerate(batch.jobs):

            for i in range(self.n_runs):  # duplicate each job, so that
                # several random initializations
                # are trialled
                job_copy = copy.deepcopy(job)
                resulting_batch.jobs.append(job_copy)

        # Return batch to send to the qpu
        return resulting_batch

    def do_post_processing(self):
        """
        Checks if the plugin needs to post process
        results.

        Returns:
            bool: the result is always True
        """
        return True

    def post_process(self, batch_result):
        """
        Perform post processing

        Args:
            batch_result (:class:`~qat.core.BatchResult`): result to post
                process

        Returns:
            :class:`~qat.core.BatchResult`
        """
        n_jobs_init = len(batch_result) // (self.n_runs)

        best_results = BatchResult()

        vals = np.reshape(
            np.array([result.value for result in batch_result.results]),
            (n_jobs_init, self.n_runs),
        )  # get all values

        for i in range(n_jobs_init):
            vals_for_job = vals[i, :]  # select row corresponding to job
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
            res_to_keep.meta_data["optimal_values_variance"] = var
            res_to_keep.meta_data["reached_values"] = vals_for_job
            best_results.results.append(res_to_keep)

        return best_results
