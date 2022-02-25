class Optimizer:
    """
    Optimizer class is a wrapper with which any user defined optimizer can be made
    compatible with the interface of VQE. The example provided below shows how to
    wrap a scipy optimizer.

    Args:
        optimization_algorithm(function): Optimization algorithm whose first argument is the
            function to minimize and second argument is initial parameters. It may have other
            parameters, either args or kwargs.
        *args_algo: valid argument for the optimization algorithm used
        **kwargs_algo: valid dictionnary for the optimization algorithm used

    Note:
        This class is there just to maintain backward compatibility.``qat.vsolve`` provides
        plugin framework to interface optimizers with quantum processors suitable for variational algorithms.

    """

    def __init__(self, optimization_algorithm, *args_algo, **kwargs_algo):
        self.algo = optimization_algorithm
        self.args_algo = args_algo
        self.kwargs_algo = kwargs_algo

    def make_calculation(self, minimized_function, initial_parameters):
        """
        This function must return 3 arguments (for now) in this order : optimised value, optimised parameters,
        number function evaluation.
        Args:
            minimized_function(function): function to minimize, it has only one parameter a numpy.array
            initial_parameters(numpy.array): initial parameters for the function to minimize

        """
        return self.algo(minimized_function, initial_parameters, *self.args_algo, **self.kwargs_algo)
