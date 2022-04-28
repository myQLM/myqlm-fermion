# -*- coding: utf-8 -*-

import pytest
import numpy as np
from qat.qpus import QPUHandler
from qat.core import Result, Job, Observable
from qat.fermion.circuits import make_shallow_circ
from qat.fermion.hamiltonians import make_embedded_model

# Define arguments to build observable
OBS_ARGS = [1, 0.5, 0.4 * np.eye(2), -0.04 * np.eye(2)]
OBS_KWARGS = {"grouping": "spins", "name": "embedded"}

try:
    from qat.core.application import Application
    from qat.generators import AbstractGenerator, AnsatzGenerator, ObservableGenerator
    from qat.plugins import TransformObservable, AbstractPlugin

except ModuleNotFoundError:
    pytest.skip(allow_module_level=True)


class FakeQPU(QPUHandler):
    "QPU checking the value of an observable or a circuit"

    def __init__(self, circ=None, obs=None):
        # Call parent constructor
        super().__init__()

        # Update attributes
        self._circ = circ
        self._obs = obs

    def submit_job(self, job):
        "Checking value of a job"
        # If circuit
        if self._circ:
            assert job.circuit.ops == self._circ.ops, "Invalid circuit"

        # If observable
        if self._obs:
            assert job.observable == self._obs, "Invalid observable"

        # Return fake result
        return Result()


class TestAnsatzGenerator:
    "Test class AnsatzGenerator"

    @staticmethod
    def test_invalid_constructor():
        "Test invalid constructor"
        # Testing with invalid method
        with pytest.raises(ValueError):
            AnsatzGenerator(name="invalid-name")

        # Testing with invalid arguments
        with pytest.raises(TypeError):
            AnsatzGenerator(name="shallow", invalid_arg=True)

    @staticmethod
    def test_plugin_mode():
        "Test plugin mode"
        stack = AnsatzGenerator(name="shallow") | FakeQPU(circ=make_shallow_circ())
        assert isinstance(stack, QPUHandler), "Stack is not a QPU"
        stack.submit(Job())  # Ensure circuit changed

        stack = TransformObservable(name="jordan-wigner") | AnsatzGenerator(name="shallow")
        assert isinstance(stack, AbstractPlugin), "Stack is not a Plugin"

        stack = AnsatzGenerator(name="shallow") | TransformObservable(name="jordan-wigner")
        assert isinstance(stack, AbstractPlugin), "Stack is not a Plugin"

    @staticmethod
    def test_generator_mode():
        "Test generator mode"
        stack = AnsatzGenerator(name="shallow") | (ObservableGenerator(*OBS_ARGS, **OBS_KWARGS) | FakeQPU(circ=make_shallow_circ()))
        assert isinstance(stack, Application), "Stack is not an Application"
        stack()

        stack = AnsatzGenerator(name="shallow") | (
            ObservableGenerator(*OBS_ARGS, **OBS_KWARGS) | TransformObservable(name="jordan-wigner")
        )
        assert isinstance(stack, AbstractGenerator)

        stack = (
            AnsatzGenerator(name="shallow")
            # Does not make sense to use these plugins in that order
            | (TransformObservable(name="jordan-wigner") | ObservableGenerator(*OBS_ARGS, **OBS_KWARGS))
        )
        assert isinstance(stack, AbstractGenerator)

    @staticmethod
    def test_serve():
        "Ensure serve method is not implemented"
        # Build generator
        generator = AnsatzGenerator(name="shallow")

        # Check serve method
        with pytest.raises(NotImplementedError):
            generator.serve()


class TestObservableGenerator:
    "Test class ObservableGenerator"

    @staticmethod
    def test_invalid_constructor():
        "Test invalid constructor"
        # Testing with no arguments
        with pytest.raises(ValueError):
            ObservableGenerator()

        # Testing with 2 observable provided
        with pytest.raises(ValueError):
            ObservableGenerator(name="embedded", observable=Observable(2))

        # Testing with invalid method
        with pytest.raises(ValueError):
            ObservableGenerator(name="invalid-name")

        # Testing with invalid arguments
        with pytest.raises(TypeError):
            ObservableGenerator(name="embedded", invalid_arg=True)

    @staticmethod
    def test_plugin_mode():
        "Test plugin mode"
        stack = ObservableGenerator(*OBS_ARGS, **OBS_KWARGS) | FakeQPU(obs=make_embedded_model(*OBS_ARGS, grouping="spins"))
        assert isinstance(stack, QPUHandler), "Stack is not a QPU"
        stack.submit(Job())  # Ensure circuit changed

        # Does not make sense to use these plugins in that order
        stack = TransformObservable(name="jordan-wigner") | ObservableGenerator(*OBS_ARGS, **OBS_KWARGS)
        assert isinstance(stack, AbstractPlugin), "Stack is not a Plugin"

        stack = ObservableGenerator(*OBS_ARGS, **OBS_KWARGS) | TransformObservable(name="jordan-wigner")
        assert isinstance(stack, AbstractPlugin), "Stack is not a Plugin"

    @staticmethod
    def test_generator_mode():
        "Test generator mode"
        stack = ObservableGenerator(*OBS_ARGS, **OBS_KWARGS) | (
            AnsatzGenerator(name="shallow") | FakeQPU(obs=make_embedded_model(*OBS_ARGS, grouping="spins"))
        )
        assert isinstance(stack, Application), "Stack is not an Application"
        stack()

        stack = ObservableGenerator(*OBS_ARGS, **OBS_KWARGS) | (
            AnsatzGenerator(name="shallow") | TransformObservable(name="jordan-wigner")
        )
        assert isinstance(stack, AbstractGenerator)

        stack = ObservableGenerator(*OBS_ARGS, **OBS_KWARGS) | (
            TransformObservable(name="jordan-wigner") | AnsatzGenerator(name="shallow")
        )
        assert isinstance(stack, AbstractGenerator)

    @staticmethod
    def test_serve():
        "Ensure serve method is not implemented"
        # Build generator
        generator = ObservableGenerator(*OBS_ARGS, **OBS_KWARGS)

        # Check serve method
        with pytest.raises(NotImplementedError):
            generator.serve()
