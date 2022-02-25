import unittest
import numpy as np
from qat.dqs.util import get_unitary_from_circuit
from qat.dqs.fermionic_util import fermionic_hamiltonian_exponential
from qat.lang.AQASM import QRoutine, X, Y, H, Z, CNOT


class TestGetUnitaryFromCircuit(unittest.TestCase):

    def setUp(self):
        self.H = np.array([[1, 1], [1, -1]]) / (2**0.5)
        self.CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.Id = np.eye(2, 2)
        self.Sz = np.array([[1, 0], [0, -1]])
        self.Sx = np.array([[0, 1], [1, 0]])
        self.Sy = np.array([[0, -1j], [1j, 0]])

    def test_get_unitary_1(self):
        q_rout = QRoutine()
        q_rout.apply(Y, 0)
        q_rout.apply(X, 1)
        q_rout.apply(H, 0)
        a = get_unitary_from_circuit(q_rout, 2)
        b = np.kron(np.dot(self.H, self.Sy), self.Sx)
        self.assertAlmostEqual(np.linalg.norm(a - b), 0, delta=0.0001)

    def test_get_unitary_2(self):
        q_rout = QRoutine()
        q_rout.apply(CNOT, 0, 1)
        q_rout.apply(CNOT, 1, 2)
        a = get_unitary_from_circuit(q_rout, 3)
        b = np.dot(np.kron(self.Id, self.CNOT), np.kron(self.CNOT, self.Id))
        self.assertAlmostEqual(np.linalg.norm(a - b), 0, delta=0.0001)

    def test_get_unitary_3(self):
        q_rout = QRoutine()
        q_rout.apply(X, 0)
        q_rout.apply(X, 1)
        q_rout.apply(H, 0)
        q_rout.apply(H, 2)
        q_rout.apply(Z, 2)
        q_rout.apply(CNOT, [1, 2])
        a = get_unitary_from_circuit(q_rout, 3)
        b = np.dot(np.kron(self.Id, self.CNOT),
                   np.dot(np.kron(self.Id, np.kron(self.Id, self.Sz)),
                          np.dot(np.kron(self.Id, np.kron(self.Id, self.H)),
                                 np.dot(np.kron(self.H, np.kron(self.Id, self.Id)),
                                        np.dot(np.kron(self.Id, np.kron(self.Sx, self.Id)),
                                               np.kron(self.Sx, np.kron(self.Id, self.Id)))))))
        self.assertAlmostEqual(np.linalg.norm(a - b), 0, delta=0.0001)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
