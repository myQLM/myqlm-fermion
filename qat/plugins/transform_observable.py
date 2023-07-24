# -*- coding: utf-8 -*-
"""
TransformObservable plugin
"""

from qat.core.plugins import AbstractPlugin
from qat.fermion.transforms import transform_to_jw_basis, transform_to_bk_basis, transform_to_parity_basis


class TransformObservable(AbstractPlugin):
    """
    Plugin performing a transformation on the Observable, to cast
    a :class:`qat.fermion.hamiltonians.FermionHamiltonian` or a :class:`qat.fermion.hamiltonians.ElectronicStructureHamiltonian`
    into a :class:`qat.fermion.hamiltonians.SpinHamiltonian`. The transformation is defined
    by a identifier (i.e. string)

     - **jordan-wigner**: Jordan-Wigner transformation
       (cf. :func:`~qat.fermion.transforms.transform_to_jw_basis`)
     - **bravyi-kitaev**: Bravyi-Kitaev transformation
       (cf. :func:`~qat.fermion.transforms.transform_to_bk_basis`)
     - **parity-basis**: Parity basis transformation
       (cf. :func:`~qat.fermion.transforms.transform_to_parity_basis`)

    Args:
        name (str): Transformation
    """

    # List of transformations
    transforms = {
        "jordan-wigner": transform_to_jw_basis,
        "bravyi-kitaev": transform_to_bk_basis,
        "parity-basis": transform_to_parity_basis,
    }

    def __init__(self, name: str):
        try:
            self.transform = TransformObservable.transforms[name]

        except KeyError as excpt:
            raise ValueError(f"Unknown transformation {name!r}") from excpt
        super().__init__()

    def compile(self, batch, specs):
        """
        Compile method. Transform every job composing a batch

        Args:
            batch (:class:`~qat.core.Batch`): batch to compile
            specs (:class:`~qat.core.HardwareSpecs`): ignored argument

        Returns:
            :class:`~qat.core.Batch`: compiled batch
        """
        # For every job
        for job in batch.jobs:
            if job.observable is not None:
                job.observable = self.transform(job.observable)

        # Return compiled batch
        return batch
