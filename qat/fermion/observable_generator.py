# -*- coding: utf-8 -*-
"""
Observable generator
"""

from qat.comm.shared.ttypes import ProcessingType
from qat.core import Job, Circuit
from qat.core.plugins import AbstractPlugin, contains_plugin
from qat.core.generator import AbstractGenerator

from .generator_tools import HybridResult
from .hamiltonians import (
    SpinHamiltonian,
    FermionHamiltonian,
    ElectronicStructureHamiltonian,
    make_hubbard_model,
    make_anderson_model,
    make_embedded_model,
)


class ObservableGenerator(AbstractPlugin, AbstractGenerator):
    """
    This ObservableGenerator class inherit from both :class:`~qat.core.generator.AbstractGenerator` and
    :class:`~qat.core.plugins.AbstractPlugin` and is designed to work with :class:`~qat.fermion.ansatz_generator.AnsatzGenerator`.

    .. run-block:: python

        import numpy as np
        from qat.generators import AnsatzGenerator, ObservableGenerator

        # Observable generation parameters
        U = 1
        mu = U / 2
        D = 0.4 * np.eye(2)
        lambda_c = -0.04 * np.eye(2)

        # VQE generators (build the same generator twice)
        generator_one = (
            AnsatzGenerator(name="shallow")
            | ObservableGenerator(U, mu, D, lambda_c, grouping="spins", name="embedded")
        )

        generator_two = (
            ObservableGenerator(U, mu, D, lambda_c, grouping="spins", name="embedded")
            | AnsatzGenerator(name="shallow")
        )

    .. warning::

        In myQLM Power Access, a remote service can't be both a remote Plugin
        and a remote BatchGenerator. If you wan to use the class as a
        BatchGenerator, please import this class using module :mod:`qlmaas.generators`,
        otherwise, please import this class using module :mod:`qlmaas.plugins`

    An :class:`~qat.fermion.observable_generator.ObservableGenerator` is build either
    from a *raw observable* or from an *observable generator*. When a generator is
    used, additional arguments and keyword arguments could be passed to build
    the observable.

    Observable builder (or Hamiltonian builders) are specified by their identifier.
    These builders are:

     - **spinhamiltonian**: Creates a :class:`~qat.fermion.hamiltonians.SpinHamiltonian`
     - **fermionhamiltonian**: Creates a :class:`~qat.fermion.hamiltonians.FermionHamiltonian`
     - **electronic-structure**: Creates a :class:`~qat.fermion.hamiltonians.ElectronicStructureHamiltonian`
     - **hubbard**: Creates a Hubbard model (cf. :func:`~qat.fermion.hamiltonians.make_hubbard_model`)
     - **anderson**: Creates a Anderson model (cf. :func:`~qat.fermion.hamiltonians.make_anderson_model`)
     - **embedded**: Creates a Embedded model (cf. :func:`~qat.fermion.hamiltonians.make_embedded_model`)

    Args:
        observable (:class:`~qat.core.Observable`, optional): raw observable
        name (str, optional): observable generator identifier
        *args: additional arguments passed to the observable generator
        **kwargs: additional keyword arguments passed to the observable generator
    """

    # List of Hamiltoninian generator
    hamiltonians = {
        "spinhamiltonian": SpinHamiltonian,
        "fermionhamiltonian": FermionHamiltonian,
        "electronic-structure": ElectronicStructureHamiltonian,
        "hubbard": make_hubbard_model,
        "anderson": make_anderson_model,
        "embedded": make_embedded_model,
    }

    def __init__(self, *args, observable=None, name: str = None, **kwargs):

        # Check arguments
        if observable is None and name is None:
            raise ValueError("An observable should be passed in argument (either argument 'observable' or 'name' should be set)")

        if observable is not None and name is not None:
            raise ValueError("Could not build the observable (argument 'observable' and 'name' are mutually exclusive)")

        # Build observable
        self.observable = observable

        if name:
            try:
                self.observable = ObservableGenerator.hamiltonians[name](*args, **kwargs)

            except KeyError as excpt:
                raise ValueError(f"Unknown hamiltonian generator {name!r}") from excpt

    def compile(self, batch, specs):
        """
        Compile a batch. This function overrides the observable (and change the job type
        into observable mde) of every job composing the batch

        Args:
            batch (:class:`~qat.core.Batch`): batch to compile
            specs (:class:`~qat.core.HardwareSpecs`): ignored argument

        Returns:
            :class:`~qat.core.Batch`: compiled batch
        """

        # For each job
        for job in batch.jobs:
            job.type = ProcessingType.OBSERVABLE
            job.observable = self.observable
            job.qubits = None

        # Return compiled batch
        return batch

    def generate(self, specs):
        """
        Generate a job - this function returns a job composed of an Ansatz
        built using the parameters of the constructor of this class

        Args:
            specs (:class:`~qat.core.HardwareSpecs`): ignored argument

        Returns:
            :class:`~qat.core.Job`: generate jobs
        """
        # Generate job
        return Job(
            None,
            circuit=Circuit(nbqbits=self.observable.nbqbits),
            observable=self.observable,
            nbshots=0,
            aggregate_data=True,
            amp_threshold=1.0 / 2**40,
        )

    def __or__(self, other):
        """
        Overrides "|" operator. If :class:`~qat.fermion.observable_generator.ObservableGenerator` is
        piped to :class:`~qat.fermion.ansatz_generator.AnsatzGenerator`,
        a :class:`~qat.core.BatchGenerator` object is returned, otherwise,
        :class:`~qat.fermion.ansatz_generator.AnsatzGenerator` is seen as a plugin
        """
        # Import ObservableGenerator here (to avoid circular import)
        # pylint: disable=import-outside-toplevel
        from .ansatz_generator import AnsatzGenerator

        # If other is an AnsatzGenerator: 'self' is seen as an BatchGenerator
        if contains_plugin(other, AnsatzGenerator):
            return AbstractGenerator.__or__(self, other)

        # Otherwise: 'self' is seen as a plugin
        return AbstractPlugin.__or__(self, other)

    def wrapper_post_process(self, obj):
        "Override wrapper_post_process method (defined for both parent class)"
        # Create hybrid result
        # pylint: disable=no-self-use
        return HybridResult(obj)

    def serve(self, *args, **kwargs):
        """
        This observable generator cannot be served (behavior not well defined). An
        exception is raised
        """
        raise NotImplementedError(
            "Method 'serve' is not implemented - ObservableGenerator "
            "is both a Plugin and a BatchGenerator, behavior of this "
            "function is not well defined"
        )
