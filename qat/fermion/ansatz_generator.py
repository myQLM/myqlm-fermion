# -*- coding: utf-8 -*-
"""
Ansatz generator
"""

from qat.core.plugins import AbstractPlugin, contains_plugin
from qat.core.generator import AbstractGenerator

from .generator_tools import HybridResult
from .circuits import (
    make_mrep_circ,
    make_mr_circ,
    make_compressed_ldca_circ,
    make_ldca_circ,
    make_general_hwe_circ,
    make_shallow_circ,
)


class AnsatzGenerator(AbstractPlugin, AbstractGenerator):
    """
    This AnsatzGenerator class inherit from both :class:`~qat.core.generator.AbstractGenerator` and
    :class:`~qat.core.plugins.AbstractPlugin` and is designed to work with
    :class:`~qat.fermion.observable_generator.ObservableGenerator`

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

    An Ansatz generator is defined by its name (i.e. a string). The supported
    Ansatz generators are:

     - **mrep**: Constructs the 8-qubit Multi-Reference Excitation Preserving (MREP) ansatz
     (cf. :func:`~qat.fermion.circuits.make_mrep_circ`).
     - **mr**: Constructs the 4-qubits Multi-Reference (MR) ansatz (cf. :func:`~qat.fermion.circuits.make_mr_circ`).
     - **compressed_ldca**: Builds a compressed version of the LDCA ansatz circuit
     (cf. :func:`~qat.fermion.circuits.make_compressed_ldca_circ`).
     - **ldca**: Constructs a LDCA circuit (cf. :func:`~qat.fermion.circuits.make_ldca_circ`).
     - **general_hwe**: Constructs an ansatz made of :math:`n_{\\mathrm{cycles}}` layers of so-called thinly-dressed routines
     (cf. :func:`~qat.fermion.circuits.make_general_hwe_circ`).
     - **shallow**: Builds a shallow ansatz (cf. :func:`~qat.fermion.circuits.make_shallow_circ`).

    Args:
        name (str): Ansatz generator identifier.
        *args: Additional arguments passed to the Ansatz generator.
        **kwargs: Additional keyword arguments passed to the Ansatz generator.

    """

    # List of Ansatz generator
    ansatz = {
        "mrep": make_mrep_circ,
        "mr": make_mr_circ,
        "compressed_ldca": make_compressed_ldca_circ,
        "ldca": make_ldca_circ,
        "general_hwe": make_general_hwe_circ,
        "shallow": make_shallow_circ,
    }

    def __init__(self, name: str, *args, **kwargs):
        # Extract generator
        try:
            self.circ = AnsatzGenerator.ansatz[name](*args, **kwargs)

        except KeyError as excpt:
            raise ValueError(f"Unknown Ansatz name {name!r}") from excpt

    def compile(self, batch, specs):
        """
        Compile a batch. This function overrides the circuit of every
        job composing the batch

        Args:
            batch (:class:`~qat.core.Batch`): batch to compile
            specs (:class:`~qat.core.HardwareSpecs`): ignored argument

        Returns:
            :class:`~qat.core.Batch`: compiled batch
        """

        # For each job
        for job in batch.jobs:
            job.circuit = self.circ

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
        return self.circ.to_job()

    def __or__(self, other):
        """
        Overrides "|" operator. If :class:`~qat.fermion.ansatz_generator.AnsatzGenerator` is
        piped to :class:`~qat.fermion.observable_generator.ObservableGenerator`,
        a :class:`~qat.core.BatchGenerator` object is returned, otherwise,
        :class:`~qat.fermion.ansatz_generator.AnsatzGenerator` is seen as a plugin
        """
        # Import ObservableGenerator here (to avoid circular import)
        # pylint: disable=import-outside-toplevel
        from .observable_generator import ObservableGenerator

        # If other is an ObservableGenerator: 'self' is seen as an BatchGenerator
        if contains_plugin(other, ObservableGenerator):
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
        This ansatz generator cannot be served (behavior not well defined). An
        exception is raised
        """

        raise NotImplementedError(
            "Method 'serve' is not implemented - AnsatzGenerator "
            "is both a Plugin and a BatchGenerator, behavior of this "
            "function is not well defined"
        )
